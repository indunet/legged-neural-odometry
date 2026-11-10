"""Isolated live simulation worker. All files stay in one unique output directory."""
import argparse
import io
import json
import os
from pathlib import Path
import time

from .common import ROOT,write_json

WIDTH,HEIGHT=1280,720
FPS=30.           # rendered frames per simulated second (streamed as MJPEG by serve.py)
STATE_PERIOD=.25  # wall-clock seconds between telemetry writes

GRID_HALF=12      # grid extends this many metres around the robot
EYE=[1.,0.,0.,0.,1.,0.,0.,0.,1.]

def add_grid(mujoco,scene,cx,cy,step=1.):
    """The camera follows the robot over a featureless floor, so draw a 1 m world-fixed grid to make motion visible."""
    import numpy as np
    x0,y0=round(cx/step)*step,round(cy/step)*step;length=GRID_HALF*step
    for i in range(-GRID_HALF,GRID_HALF+1):
        major=(round(i+y0/step)%5==0,round(i+x0/step)%5==0)
        for axis in (0,1):
            if scene.ngeom>=scene.maxgeom:return
            along=major[axis]  # lines every 5 m are brighter and thicker
            width=.018 if along else .008;alpha=.55 if along else .28
            size=[length,width,.0004] if axis==0 else [width,length,.0004]
            pos=[x0,y0+i*step,.003] if axis==0 else [x0+i*step,y0,.003]
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_BOX,np.array(size),np.array(pos),
                                np.array(EYE),np.array([.85,.92,1.,alpha],dtype=np.float32))
            scene.ngeom+=1

HEADING_GAIN=2.   # rad/s of yaw-rate correction per rad of heading error
HEADING_LIMIT=.4  # rad/s

class HeadingHold:
    """Demo-side yaw lock. The policy tracks a velocity command open loop and drifts about 1 deg/s in yaw,
    so a straight-ahead command walks a wide circle. While walking with wz = 0, steer back to the heading
    held when the turn command was released. Only the simulated driver uses this; the odometry never sees it."""
    def __init__(self):self.target=None

    def __call__(self,command,yaw):
        import numpy as np
        command=np.asarray(command,float)
        if abs(command[2])>1e-3 or np.linalg.norm(command)<.1:
            self.target=None;return command
        if self.target is None:self.target=yaw
        error=(self.target-yaw+np.pi)%(2*np.pi)-np.pi
        drive=command.copy();drive[2]=float(np.clip(HEADING_GAIN*error,-HEADING_LIMIT,HEADING_LIMIT));return drive

def run(folder):
    folder=Path(folder);config=json.loads((folder/'config.json').read_text())
    resource=ROOT/'configs/resources.json'
    gpu=json.loads(resource.read_text()).get('render_gpu',6) if resource.exists() else 6
    os.environ['MUJOCO_GL']='egl';os.environ['MUJOCO_EGL_DEVICE_ID']=str(gpu)
    os.environ['PYOPENGL_PLATFORM']='egl'
    import mujoco
    import numpy as np
    from PIL import Image
    from .simulation import Simulation
    from .runtime import Odometry
    sim=Simulation(seed=723,noise=True)
    # Explicit local model override is for CLI validation, not exposed by the HTTP API.
    model=ROOT/config['model'] if config.get('model') else None
    if model is not None:model.resolve().relative_to(ROOT)
    estimator=Odometry(model=model)
    sim.model.vis.global_.offwidth=WIDTH;sim.model.vis.global_.offheight=HEIGHT
    renderer=mujoco.Renderer(sim.model,height=HEIGHT,width=WIDTH)
    camera=mujoco.MjvCamera();camera.distance=3.3;camera.azimuth=135;camera.elevation=-24
    heading=HeadingHold();command=np.zeros(3);paused=False;rows=[];origin=None;control_mtime=0
    last_wall=time.monotonic();last_sim=0.;last_render=-1.;last_frame_wall=0.;last_state_wall=0.
    try:
        while not (folder/'stop').exists():
            control=folder/'control.json'
            if control.exists() and control.stat().st_mtime_ns!=control_mtime:
                control_mtime=control.stat().st_mtime_ns
                try:
                    msg=json.loads(control.read_text());command=np.array(msg.get('command',command))
                    paused=bool(msg.get('paused',paused))
                    if msg.get('reset'):
                        sim.reset();estimator.reset();heading.target=None;rows=[];origin=None;last_render=-1.;last_state_wall=0.
                        last_wall=time.monotonic();last_sim=0.
                except (ValueError,OSError):pass
            if not paused and not sim.fallen():
                sim.step(heading(command,sim.truth()[1][3]));result=estimator.update(sim.observe(imu_yaw=True))
                if result:
                    velocity,truth=sim.truth()
                    if origin is None:origin=truth.copy()
                    delta=truth[:2]-origin[:2];c,s=np.cos(origin[3]),np.sin(origin[3])
                    actual=[float(c*delta[0]+s*delta[1]),float(-s*delta[0]+c*delta[1]),
                            float(np.arctan2(np.sin(truth[3]-origin[3]),np.cos(truth[3]-origin[3])))]
                    cov=result['pose_cov']
                    sigma=float(np.sqrt(cov[0]+cov[4])) if cov else None
                    speed=result['uncertainty']['epistemic_speed_std'] if result['uncertainty'] else None
                    rows.append({'t':result['t'],'pose':result['pose'],'truth':actual,'velocity':result['velocity'],
                                 'truth_velocity':velocity[:2].tolist(),'stationary':result['stationary'],
                                 'sigma_xy':sigma,'epistemic_speed_std':speed,'quality':result['quality']})
                    rows=rows[-10000:]
            now=time.monotonic()
            if sim.data.time-last_render>=1/FPS or ((paused or sim.fallen()) and now-last_frame_wall>=.2):
                camera.lookat[:]=sim.data.xpos[sim.base]
                renderer.update_scene(sim.data,camera=camera)
                add_grid(mujoco,renderer.scene,*sim.data.xpos[sim.base][:2])
                rgb=renderer.render();image=Image.fromarray(rgb);buf=io.BytesIO();image.save(buf,format='JPEG',quality=80)
                tmp=folder/'frame.tmp.jpg';tmp.write_bytes(buf.getvalue());tmp.replace(folder/'frame.jpg')
                last_render=sim.data.time;last_frame_wall=now
            if now-last_state_wall>=STATE_PERIOD:
                write_json(folder/'state.json',{'status':'fallen' if sim.fallen() else 'paused' if paused else 'running',
                           'calibrated':estimator.builder.attitude.ready,
                           'heading_reference':'理想 IMU 姿态仿真（非真机精度）',
                           'samples':rows,'command':command.tolist(),'updated':time.time()})
                last_state_wall=now
            target=last_wall+(sim.data.time-last_sim)
            if paused or sim.fallen():
                time.sleep(.05);last_wall=time.monotonic();last_sim=sim.data.time
            else:time.sleep(max(0,min(.02,target-time.monotonic())))
    finally:renderer.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args()
    try:run(a.folder)
    except Exception as error:
        import traceback
        write_json(Path(a.folder)/'state.json',{'status':'error','error':repr(error),'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
