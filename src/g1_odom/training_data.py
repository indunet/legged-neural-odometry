"""Dataset loader and constants consumed by the training recipe in ``trainer``."""
import json
from pathlib import Path

import numpy as np

from .common import FEATURE_NAMES, JOINTS  # noqa: F401  (re-exported to trainer)

FK_FEATURES=()
FRAME_RATE=50.
JOINT_RATE=50.
LABEL_NAMES=('vx','vy')
LABEL_TAU=.2
NO_ANKLE_JOINTS=JOINTS
WINDOW=25


def label_columns(stored_names,outputs):
    return tuple(tuple(stored_names).index(name) for name in outputs)


def load_dataset(path,trim_falls=True):
    """Episodes with at least one window; falls lose their final second."""
    path=Path(path)
    manifest=json.loads((path/'manifest.json').read_text()) if (path/'manifest.json').exists() else None
    files=[(path/e['file'],e) for e in manifest['episodes'] if e.get('file') and e.get('frames',0)] if manifest else [(p,None) for p in sorted(path.glob('ep_*.npz'))]
    rows=[]
    for file,_ in files:
        with np.load(file) as f:
            info=json.loads(str(f['info']))
            stop=len(f['time'])-(50 if trim_falls and info['fell'] else 0)
            r={k:f[k][:max(0,stop)].copy() for k in ('features','label','pose','time')}
            r['config']=info['config'];r['info']=info
            r['label_names']=np.array(['vx','vy','vz'])
        r['info']['file']=str(file.resolve())
        if len(r['features'])>=WINDOW:
            if not np.isfinite(r['features']).all() or not np.isfinite(r['label']).all():raise ValueError(f'nonfinite data: {file}')
            rows.append(r)
    return rows
