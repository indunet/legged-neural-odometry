"""G1-only planar dataset: command/stop schedules on the G1 flat policy."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import mujoco
import numpy as np

from . import planar_conditions as C
from .common import ROOT,DT,FRAME_DT,FEATURE_NAMES,write_json,sha256
from .sensors import FeatureBuilder
from .simulation import Simulation,FLAT_POLICY

GENERATION='g1_planar'
DATA=ROOT/'data'/GENERATION
COUNTS={'train':2000,'stop_train':600,'val':200,'stop_val':60,
        'test_iid':100,'test_ood':60,'odometry':30,'stop_test':60}
TRAIN_SPLITS=('train','stop_train')
VAL_SPLITS=('val','stop_val')
TEST_SPLITS=('test_iid','test_ood','odometry','stop_test')


def configs(pilot=False):
    jobs=[]
    for group,(split,count) in enumerate(COUNTS.items()):
        if pilot:count=min(count,8 if split=='train' else 2)
        for index in range(count):
            seed=400000000+group*1000000+index
            rng=np.random.default_rng(seed)
            duration=90. if split=='odometry' else float(rng.uniform(24,40))
            schedule=C.stop_schedule(rng,duration) if split.startswith('stop') else C.sample_schedule(rng,duration)
            # Keep a full stationary calibration interval before locomotion.
            shift=max(0.,2.-schedule[0]['t1']);schedule[0]['t1']+=shift
            for segment in schedule[1:]:
                segment['t0']+=shift;segment['t1']+=shift
            nominal=split=='odometry'
            if nominal:
                table=[(.4,0,0),(-.25,0,0),(0,.2,0),(0,-.2,0),(.3,0,.35),
                       (.3,0,-.35),(0,0,.4),(0,0,0),(.5,.1,0),(.2,-.1,.2)]
                schedule=[{'t0':0.,'t1':2.,'mode':'stand','target':[0.,0.,0.],'transition':'step','ramp':0.},
                          {'t0':2.,'t1':duration,'mode':'long_'+str(index%10),'target':list(table[index%10]),'transition':'step','ramp':0.}]
            ood=split=='test_ood'
            physics={'friction':.8 if nominal else float(rng.uniform(.18,.25) if ood else rng.uniform(.3,1.2)),
                     'mass_scale':1. if nominal else float(rng.uniform(.9,1.1)),
                     'payload':0. if nominal else float(rng.uniform(3.5,5.) if ood else rng.uniform(0.,3.)),
                     'kp_scale':(np.ones(29) if nominal else rng.uniform(.7,.85,29) if ood else rng.uniform(.85,1.15,29)).tolist(),
                     'kd_scale':(np.ones(29) if nominal else rng.uniform(.85,1.15,29)).tolist(),
                     'torque_scale':1. if nominal else float(rng.uniform(.7,.8) if ood else rng.uniform(.8,1.))}
            sensors={'gyro_std':.002 if nominal else float(rng.uniform(.001,.004)),
                     'accel_std':.04 if nominal else float(rng.uniform(.02,.08)),
                     'joint_position_std':.0003 if nominal else float(rng.uniform(.0001,.0008)),
                     'joint_velocity_std':.008 if nominal else float(rng.uniform(.004,.012)),
                     'gyro_walk_std':.00015 if nominal else float(rng.uniform(.0001,.0005))}
            pushes=[];yaw_torques=[]
            if not nominal and rng.random()<.6:
                for t in np.arange(float(rng.uniform(5.,10.)),duration-2.,float(rng.uniform(7.,12.))):
                    angle=float(rng.uniform(-np.pi,np.pi));force=float(rng.uniform(20.,80.))
                    pushes.append({'t':float(t),'duration':float(rng.uniform(.1,.35)),
                                   'force':[force*np.cos(angle),force*np.sin(angle),0.]})
            if not nominal and rng.random()<.25:
                yaw_torques=[{'t':float(rng.uniform(7.,15.)),'duration':float(rng.uniform(.1,.3)),
                              'torque':float(rng.uniform(-8.,8.))}]
            jobs.append({'generation':GENERATION,'split':split,'index':index,'seed':seed,
                         'duration':duration,'policy':'flat','course':{'kind':'flat'},'physics':physics,
                         'sensors':sensors,'schedule':schedule,'mismatch':C.nominal_mismatch() if nominal else C.sample_mismatch(rng),
                         'pushes':pushes,'yaw_torques':yaw_torques,'version':1})
    return jobs


def configure_sim(job):
    sim=Simulation(job['seed'],noise=True,randomize=False)
    p=job['physics'];sim.model.geom_friction[:,0]=p['friction']
    sim.model.body_mass[1:]*=p['mass_scale'];sim.model.body_inertia[1:]*=p['mass_scale']
    sim.model.body_mass[sim.model.body('torso_link').id]+=p['payload']
    sim.kp_scale=np.array(p['kp_scale']);sim.kd_scale=np.array(p['kd_scale'])
    sim.policy.limits*=p['torque_scale']
    for name,value in job['sensors'].items():setattr(sim,name,value)
    mujoco.mj_setConst(sim.model,sim.data);sim.reset()
    return sim


def generate(job,output):
    output=Path(output);folder=output/job['split'];folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"ep_{job['index']:05d}.npz"
    if path.exists():
        with np.load(path) as f:info=json.loads(str(f['info']))
        if info['config']!=job:raise ValueError(f'existing episode config differs: {path}')
        return info
    begin=time.monotonic();sim=configure_sim(job);builder=FeatureBuilder()
    schedule=C.CommandSchedule(job,np.random.default_rng(job['seed']+77))
    filtered=np.zeros(3);rows={key:[] for key in ('features','label','pose','time','heading','imu_yaw','calibrated','command')}
    for _ in range(int(job['duration']/DT)):
        t=float(sim.data.time);request,_=schedule.request(t,DT);command=schedule.policy_command(request,t)
        push=np.zeros(3);torque=0.
        for event in job['pushes']:
            if event['t']<=t<event['t']+event['duration']:push+=event['force']
        for event in job['yaw_torques']:
            if event['t']<=t<event['t']+event['duration']:torque+=event['torque']
        sim.data.xfrc_applied[sim.base,5]=torque
        sim.step(command,push);sample=sim.observe(imu_yaw=True);frame=builder.push(sample)
        if frame is not None:
            velocity,pose=sim.truth();filtered+=FRAME_DT/(.2+FRAME_DT)*(velocity-filtered)
            for name,value in {'features':frame,'label':filtered.copy(),'pose':pose,'time':sample.t,
                               'heading':builder.attitude.yaw,'imu_yaw':sample.imu_yaw,
                               'calibrated':builder.attitude.ready,'command':command}.items():rows[name].append(value)
        if sim.fallen():break
    info={'generation':GENERATION,'robot':'Unitree G1 29-DOF','config':job,'file':str(path.relative_to(output)),
          'frames':len(rows['time']),'seconds':float(sim.data.time),'fell':sim.fallen(),'finished':not sim.fallen(),
          'policy':'flat','calibrated':builder.attitude.ready,'wall_seconds':time.monotonic()-begin,
          'imu_yaw_reference':'ideal simulated IMU attitude output; not real Unitree hardware accuracy'}
    arrays={name:np.asarray(value,dtype=np.float32) for name,value in rows.items()}
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**arrays,info=json.dumps(info),feature_names=np.array(FEATURE_NAMES))
    temp.replace(path);return info


def worker(args):
    job,output=args
    try:return generate(job,output)
    except Exception as error:
        import traceback
        return {'config':job,'error':repr(error),'traceback':traceback.format_exc()}


def collect(pilot=False,workers=8):
    output=ROOT/'data'/(GENERATION+'_pilot' if pilot else GENERATION);jobs=configs(pilot)
    output.mkdir(parents=True,exist_ok=True)
    config_hash=hashlib.sha256(json.dumps(jobs,sort_keys=True).encode()).hexdigest()
    write_json(output/'config.json',{'generation':GENERATION,'counts':COUNTS,'pilot':pilot,'config_sha256':config_hash,
              'robot':'G1 only','policy_standing_gate':True,'reuse_historical_data':False,'frame_features':FEATURE_NAMES,
              'conditions':'continuous command/stop schedules; G1-specific flat physics and sensor randomization',
              'source_code_sha256':sha256(Path(__file__)),'policy_sha256':sha256(FLAT_POLICY)})
    results=[]
    def save():write_json(output/'manifest.json',{'generation':GENERATION,'config_sha256':config_hash,
                      'expected':len(jobs),'completed':len(results),'episodes':results})
    with ProcessPoolExecutor(max_workers=min(8,workers),mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,(job,str(output))) for job in jobs]
        for future in as_completed(futures):
            row=future.result();results.append(row)
            if len(results)%10==0 or len(results)==len(jobs):save()
            write_json(ROOT/'runs/data_progress.json',{'stage':'g1_data','output':str(output),
                       'done':len(results),'total':len(jobs),'last':row,'updated':time.time()})
            print(len(results),'/',len(jobs),row.get('file'),row.get('fell'),row.get('error'),flush=True)
    save()
    if any(r.get('error') for r in results):raise RuntimeError('G1 generation errors; see manifest')
    return output


def audit(splits=None):
    """Check every expected file against its config and feature contract; hash them.

    Returns the report and the SHA256 of the per-file hash list, which pins the
    training data in model lineage.
    """
    splits=set(splits or COUNTS);errors=[];digests={};groups={s:[] for s in splits}
    for job in configs():
        if job['split'] not in splits:continue
        relative=f"{job['split']}/ep_{job['index']:05d}.npz";path=DATA/relative
        if not path.exists():errors.append(f'missing {relative}');continue
        with np.load(path) as f:
            info=json.loads(str(f['info']))
            if info['config']!=job or info.get('generation')!=GENERATION:errors.append(f'config {relative}')
            if job['split'] in TRAIN_SPLITS and not info['calibrated']:errors.append(f'uncalibrated {relative}')
            if tuple(f['feature_names'])!=FEATURE_NAMES or f['features'].shape[1:]!=(25,):errors.append(f'features {relative}')
            n=len(f['time'])
            if not n or any(len(f[k])!=n for k in ('features','label','pose','imu_yaw')):errors.append(f'lengths {relative}')
            if n>1 and not np.allclose(np.diff(f['time']),FRAME_DT,atol=2e-5):errors.append(f'cadence {relative}')
            groups[job['split']].append({'seed':job['seed'],'fell':info['fell'],'calibrated':info['calibrated']})
        digests[relative]=sha256(path)
    seeds=[r['seed'] for rows in groups.values() for r in rows]
    if len(seeds)!=len(set(seeds)):errors.append('overlapping trajectory seeds')
    report={'pass':not errors,'data_root':str(DATA),'errors':errors[:20],
            'splits':{s:{'episodes':len(r),'falls':sum(x['fell'] for x in r),
                         'uncalibrated':sum(not x['calibrated'] for x in r)} for s,r in groups.items()}}
    encoded=json.dumps(digests,sort_keys=True).encode()
    report['files_sha256']=hashlib.sha256(encoded).hexdigest()
    if errors:raise RuntimeError(f'data audit failed: {errors[:5]}')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();collect(a.pilot,a.workers)


if __name__=='__main__':main()
