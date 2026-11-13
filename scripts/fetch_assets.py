#!/usr/bin/env python3
"""Check the vendored G1 model and walk policy against configs/assets.lock.json.

The files live in assets/unitree_g1/ and are part of this repository. This script
no longer clones unitree_rl_mjlab.
"""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT/'configs/assets.lock.json'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='accepted for compatibility; verification always runs')
    parser.parse_args()
    lock = json.loads(LOCK.read_text())
    files = lock.get('files') or []
    if not files:
        raise SystemExit(f'{LOCK} has no files')
    for item in files:
        path = ROOT/item['path']
        if not path.is_file():
            raise SystemExit(f'missing {path}')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != item['sha256'] or len(data) != item['bytes']:
            raise SystemExit(f'{path} does not match the locked hash')
    print(f'Verified {len(files)} asset files')

if __name__ == '__main__':
    main()
