"""Isolated live simulation worker. All files stay in one unique output directory."""
import argparse
import io
import json
import os
from pathlib import Path
import time

from .common import ROOT,write_json

WIDTH,HEIGHT=1280,720
FPS=15.           # rendered frames per wall-clock second (streamed as MJPEG by serve.py)
RENDER_SLACK=.03  # only render when the simulation is at most this many seconds behind real time
STATE_PERIOD=.25  # wall-clock seconds between telemetry writes
MAX_LAG=.5        # wall-clock seconds the simulation may fall behind before it stops trying to catch up
RECENT=750        # telemetry rows sent to the browser (15 s at 50 Hz), enough for the time-series charts
TRACK_POINTS=2000 # the full-run trajectory is thinned by half whenever it grows past this many points

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

def render_config():
    """CPU software rendering unless resources.json or MUJOCO_GL says otherwise.
    Shadows and MSAA are off by default: under osmesa/llvmpipe they cost about 250 ms per 1280x720 frame."""
    resource=ROOT/'configs/resources.json'
    cfg=json.loads(resource.read_text()) if resource.exists() else {}
    backend=os.environ.get('MUJOCO_GL') or cfg.get('render_gl') or 'osmesa'
    os.environ['MUJOCO_GL']=backend
    if backend=='egl':
        os.environ['MUJOCO_EGL_DEVICE_ID']=str(cfg.get('render_gpu',0))
        os.environ['PYOPENGL_PLATFORM']='egl'
    elif backend=='osmesa':
        os.environ['PYOPENGL_PLATFORM']='osmesa'
    width=int(cfg.get('render_width',WIDTH))
    height=int(cfg.get('render_height',HEIGHT))
    fps=float(cfg.get('render_fps',FPS))
    shadows=bool(cfg.get('render_shadows',False))
    return width,height,fps,shadows

def run(folder):
    folder=Path(folder);config=json.loads((folder/'config.json').read_text())
    width,height,fps,shadows=render_config()
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
    sim.model.vis.global_.offwidth=width;sim.model.vis.global_.offheight=height
    if not shadows:sim.model.vis.quality.offsamples=0
    renderer=mujoco.Renderer(sim.model,height=height,width=width)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=shadows
    camera=mujoco.MjvCamera();camera.distance=3.3;camera.azimuth=135;camera.elevation=-24
    heading=HeadingHold();command=np.zeros(3);paused=False;rows=[];track=[];stride=1;seen=0;origin=None;control_mtime=0
    last_wall=time.monotonic();last_sim=0.;next_frame=0.;last_state_wall=0.
    try:
        while not (folder/'stop').exists():
            control=folder/'control.json'
            if control.exists() and control.stat().st_mtime_ns!=control_mtime:
                control_mtime=control.stat().st_mtime_ns
                try:
                    msg=json.loads(control.read_text());command=np.array(msg.get('command',command))
                    paused=bool(msg.get('paused',paused))
                    if msg.get('reset'):
                        sim.reset();estimator.reset();heading.target=None;rows=[];track=[];stride=1;seen=0;origin=None
                        next_frame=0.;last_state_wall=0.;last_wall=time.monotonic();last_sim=0.
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
                    r5=lambda values:[round(float(v),5) for v in values]
                    rows.append({'t':round(result['t'],3),'pose':r5(result['pose']),'truth':r5(actual),
                                 'velocity':r5(result['velocity']),'truth_velocity':r5(velocity[:2]),
                                 'stationary':result['stationary'],'sigma_xy':None if sigma is None else round(sigma,6),
                                 'epistemic_speed_std':None if speed is None else round(float(speed),6),'quality':result['quality']})
                    rows=rows[-RECENT:]
                    if seen%stride==0:
                        track.append([round(v,4) for v in (actual[0],actual[1],result['pose'][0],result['pose'][1])])
                        if len(track)>TRACK_POINTS:track=track[::2];stride*=2
                    seen+=1
            now=time.monotonic()
            idle=paused or sim.fallen()
            behind=now-(last_wall+(sim.data.time-last_sim))
            if now>=next_frame and (idle or behind<=RENDER_SLACK):
                camera.lookat[:]=sim.data.xpos[sim.base]
                renderer.update_scene(sim.data,camera=camera)
                rgb=renderer.render();image=Image.fromarray(rgb);buf=io.BytesIO();image.save(buf,format='JPEG',quality=80)
                tmp=folder/'frame.tmp.jpg';tmp.write_bytes(buf.getvalue());tmp.replace(folder/'frame.jpg')
                next_frame=now+(.2 if idle else 1/fps)
            if now-last_state_wall>=STATE_PERIOD:
                write_json(folder/'state.json',{'status':'fallen' if sim.fallen() else 'paused' if paused else 'running',
                           'calibrated':estimator.builder.attitude.ready,
                           'heading_reference':'理想 IMU 姿态仿真（非真机精度）',
                           'samples':rows,'track':track,'command':command.tolist(),'updated':time.time()})
                last_state_wall=now
            target=last_wall+(sim.data.time-last_sim)
            if idle or time.monotonic()-target>MAX_LAG:
                if idle:time.sleep(.05)
                last_wall=time.monotonic();last_sim=sim.data.time
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
