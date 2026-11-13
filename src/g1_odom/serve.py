"""Local standalone Web console; no original-project imports or services."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from .common import ROOT,write_json
from .runtime import MODEL

WEB=Path(__file__).resolve().parent/'web'
BOUNDARY='g1frame'

def read_json(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (ValueError,OSError):return {} if default is None else default

class App:
    def __init__(self):
        self.lock=threading.RLock();self.proc=None;self.folder=None;self.job=None

    def stop(self):
        with self.lock:
            if self.folder:(self.folder/'stop').touch()
            if self.proc and self.proc.poll() is None:
                try:self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
                    try:self.proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
            self.proc=None
            if self.folder:
                state=read_json(self.folder/'state.json')
                state.update(status='stopped',updated=time.time())
                write_json(self.folder/'state.json',state)

    def start(self,body):
        if not MODEL.is_file():raise ValueError('正式模型尚未发布，请在训练页查看进度')
        with self.lock:
            self.stop();self.folder=ROOT/'outputs/live'/uuid.uuid4().hex
            self.folder.mkdir(parents=True)
            write_json(self.folder/'config.json',{})
            log=open(self.folder/'worker.log','w')
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            self.proc=subprocess.Popen([sys.executable,'-m','g1_odom.live',str(self.folder)],cwd=ROOT,
                                       env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close()
        return {'ok':True}

    def control(self,body):
        if not self.folder or not self.proc or self.proc.poll() is not None:raise ValueError('仿真尚未启动')
        current=read_json(self.folder/'control.json',{'command':[0,0,0],'paused':False})
        current.pop('reset',None)
        if 'command' in body:
            cmd=body['command']
            if not isinstance(cmd,list) or len(cmd)!=3:raise ValueError('command must contain vx, vy, wz')
            limits=((-0.5,1.),(-.5,.5),(-1.,1.))
            if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in cmd):
                raise ValueError('command must contain finite numbers')
            current['command']=[min(high,max(low,float(x))) for x,(low,high) in zip(cmd,limits)]
        if 'paused' in body:current['paused']=bool(body['paused'])
        if body.get('reset'):current['reset']=True
        write_json(self.folder/'control.json',current);return {'ok':True}

    def status(self):
        state=read_json(self.folder/'state.json') if self.folder else {'status':'idle','samples':[]}
        if self.proc and self.proc.poll() is None and not state:
            state={'status':'starting','samples':[]}
        if self.proc and self.proc.poll() is not None and state.get('status') not in ('error','stopped'):
            state={'status':'error','error':f'worker exited {self.proc.returncode}','samples':[]}
        state['released']=MODEL.is_file()
        state['session']=self.folder.name if self.folder else None
        state['pipeline']=read_json(ROOT/'runs/progress.json')
        data=read_json(ROOT/'runs/data_progress.json');data.pop('last',None);state['data']=data
        return state

    def start_job(self):
        from .workflow import active_jobs
        running=active_jobs()
        if running:return {'ok':True,'message':'采样或训练流程正在运行','pids':running}
        if self.job and self.job.poll() is None:return {'ok':True,'message':'任务正在运行'}
        log=open(ROOT/'runs/workflow.log','a')
        device=read_json(ROOT/'configs/resources.json').get('device','cpu')
        self.job=subprocess.Popen([sys.executable,'-m','g1_odom','run','--device',device],cwd=ROOT,stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True,
                                  env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        log.close();return {'ok':True,'pid':self.job.pid}

def report_summary():
    """Headline accuracy and uncertainty of the released model from reports/evaluation.json."""
    report=read_json(ROOT/'reports/evaluation.json')
    if not report.get('datasets'):return {}
    cal=report.get('calibration') or {}
    splits={}
    for name,d in report['datasets'].items():
        o=d['released']['overall'];u=d.get('uncertainty') or {}
        splits[name]={'episodes':o['episodes'],'fell':o['fell'],'endpoint_xy_m':o['endpoint_xy_m'],'ate_xy_m':o['ate_xy_m'],
                      'rpe_5m':o['rpe']['5']['position_mean_m'],
                      'no_lag_endpoint_mean_m':d['no_lag_compensation']['overall']['endpoint_xy_m']['mean'],
                      'coverage_1sigma':u.get('coverage_1sigma'),'coverage_2sigma':u.get('coverage_2sigma'),
                      'pose_coverage_2sigma':u.get('pose_coverage_2sigma'),'pose_nees_mean':u.get('pose_nees_mean'),
                      'warn_or_alarm':(u.get('quality_fraction') or {}).get('warn_or_alarm')}
    return {'sha256':report.get('locked_sha256'),'reference':report.get('heading_reference'),'datasets':splits,
            'calibration':{k:cal.get(k) for k in ('aleatoric_scale','epistemic_scale','correlation_time_s','bias_fraction','pose_scale')},
            'detection':report.get('detection'),'online':(report.get('online') or {}).get('scenarios')}

APP=App()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def respond(self,data,status=200,mime='application/json'):
        value=json.dumps(data,ensure_ascii=False).encode() if mime=='application/json' else data
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(value)));self.end_headers()
        try:self.wfile.write(value)
        except (BrokenPipeError,ConnectionResetError):pass

    def stream(self):
        """Motion JPEG: push every new rendered frame; repeat the last one as a keep-alive so dead clients are noticed."""
        self.send_response(200);self.send_header('Content-Type',f'multipart/x-mixed-replace; boundary={BOUNDARY}')
        self.send_header('Cache-Control','no-store');self.send_header('Connection','close');self.end_headers()
        last=None;data=None;sent=0.
        try:
            while True:
                image=APP.folder/'frame.jpg' if APP.folder else None
                try:
                    stat=image.stat() if image else None
                    key=(stat.st_mtime_ns,stat.st_size) if stat else None
                    if key and key!=last:data=image.read_bytes();last=key;sent=0.
                except OSError:pass
                if data and time.monotonic()-sent>=2.:
                    self.wfile.write(f'--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n'.encode()+data+b'\r\n')
                    self.wfile.flush();sent=time.monotonic()
                time.sleep(.01)
        except (BrokenPipeError,ConnectionResetError,OSError):pass

    def do_GET(self):
        path=self.path.split('?')[0]
        if path=='/api/state':return self.respond(APP.status())
        if path=='/api/stream':return self.stream()
        if path=='/api/frame':
            image=APP.folder/'frame.jpg' if APP.folder else None
            if image and image.is_file():return self.respond(image.read_bytes(),mime='image/jpeg')
            return self.respond({'error':'no frame'},404)
        if path=='/api/reports':return self.respond(report_summary())
        if path in ('/','/odometry','/training'):return self.respond((WEB/'index.html').read_bytes(),mime='text/html; charset=utf-8')
        if path in ('/app.js','/style.css'):
            return self.respond((WEB/path[1:]).read_bytes(),mime='application/javascript' if path.endswith('.js') else 'text/css')
        self.respond({'error':'not found'},404)

    def do_POST(self):
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<=length<=16384:raise ValueError('request too large')
            body=json.loads(self.rfile.read(length) or b'{}')
            if not isinstance(body,dict):raise ValueError('JSON object expected')
            if self.path=='/api/sim/start':return self.respond(APP.start(body))
            if self.path=='/api/sim/control':return self.respond(APP.control(body))
            if self.path=='/api/sim/stop':APP.stop();return self.respond({'ok':True})
            if self.path=='/api/training/start':return self.respond(APP.start_job())
            if self.path=='/api/training/stop':
                from .workflow import stop_jobs
                return self.respond(stop_jobs())
            self.respond({'error':'not found'},404)
        except (ValueError,RuntimeError) as error:self.respond({'error':str(error)},400)

def main():
    (ROOT/'runs').mkdir(exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=18078)
    args=p.parse_args();server=ThreadingHTTPServer((args.host,args.port),Handler)
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    server.daemon_threads=True
    print(f'G1 odometry: http://{args.host}:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:APP.stop();server.server_close()

if __name__=='__main__':main()
