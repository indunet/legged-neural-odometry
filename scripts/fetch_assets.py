#!/usr/bin/env python3
"""Reproduce the pinned public policy/model downloads under this project only."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'unitree_rl_mjlab':{'url':'https://github.com/unitreerobotics/unitree_rl_mjlab.git',
                     'commit':'1425b15f73bd4095f0df53709d7c389c3eb9e790',
                     'sparse':['src/assets/robots/unitree_g1','deploy/robots/g1/config/policy/velocity/v0','src/tasks/velocity/config/g1'],
                     'license':'No root LICENSE supplied in the pinned upstream checkout'}
}

def command(args,cwd=ROOT):subprocess.run(args,cwd=cwd,check=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
 (ROOT/'third_party').mkdir(exist_ok=True)
 for name,source in SOURCES.items():
  dest=ROOT/'third_party'/name
  if not dest.exists():
   if args.verify:raise SystemExit(f'missing {dest}')
   command(['git','clone','--filter=blob:none','--no-checkout',source['url'],str(dest)])
   if source.get('sparse'):command(['git','sparse-checkout','set',*source['sparse']],dest)
   command(['git','checkout',source['commit']],dest)
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=dest,text=True).strip()
  if commit!=source['commit']:raise SystemExit(f'{name}: checkout differs from pinned commit')
 files=[]
 for base in (ROOT/'third_party/unitree_rl_mjlab/src/assets/robots/unitree_g1',
              ROOT/'third_party/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0'):
  for f in sorted(base.rglob('*')):
   if f.is_file():files.append({'path':str(f.relative_to(ROOT)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 manifest={'sources':SOURCES,'files':files}
 out=ROOT/'configs/assets.lock.json';out.write_text(json.dumps(manifest,indent=2)+'\n')
 print(f'Verified {len(files)} asset files')

if __name__=='__main__':main()
