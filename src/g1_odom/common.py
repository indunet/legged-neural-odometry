from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]  # repository root (src layout; install with `pip install -e .`)
DT = 0.002
FRAME_DT = 0.02
JOINTS = tuple(f'{side}_{joint}_joint' for side in ('left', 'right')
               for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee'))
FEATURE_NAMES = tuple([f'gyro_{a}' for a in 'xyz'] + [f'accel_{a}' for a in 'xyz']
                      + [f'gravity_{a}' for a in 'xyz']
                      + [f'q_{j}' for j in JOINTS] + [f'dq_{j}' for j in JOINTS])

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)

def gpu_device():
    import subprocess
    import torch
    if not torch.cuda.is_available():
        return 'cpu'
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True)
    available = [(int(free), int(idx)) for idx, free in (r.split(',') for r in rows.splitlines())]
    choices = [(free,idx) for free,idx in available if idx in (4,5,6,7)] or available
    free, idx = max(choices)
    if free < 4096:
        raise RuntimeError('No GPU among 4–7 has 4 GiB free; existing jobs are left running')
    return f'cuda:{idx}'
