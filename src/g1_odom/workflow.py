"""Full workflow: collect -> audit -> label audit -> train -> calibrate -> evaluate; every stage resumes."""
import argparse
import os
import signal

import numpy as np

from .common import ROOT, write_json
from .train import REPORT, RUN, TRAIN_SPLITS, VAL_SPLITS, progress

STAGES = ('collect', 'audit', 'labels', 'train', 'calibrate', 'evaluate')


def label_audit():
    """Perfect-label floor on training/validation splits: labels integrated as if the network were exact."""
    from .dataset import DATA
    from .evaluate import integrate, relative_pose
    from .labels import lag_correction, lag_gain, world_lowpass_labels
    from .training_data import load_dataset
    progress(status='label_audit')
    report = {'note': 'labels integrated as if the network were perfect; positions against the true pelvis pose',
              'splits': {}}
    for split in (*TRAIN_SPLITS, *VAL_SPLITS):
        heading_frame, world, frames = [], [], 0
        for row in load_dataset(DATA/split):
            t = row['time']
            ends = np.arange(max(24, int(np.searchsorted(t, 1.5))), len(t))
            if len(ends) < 3:
                continue
            yaw = np.unwrap(row['pose'][:, 3])
            truth = relative_pose(np.c_[row['pose'][ends, :3], yaw[ends]])
            stored = integrate(row['label'][ends, :2], yaw[ends])
            label = world_lowpass_labels(row['pose'], t)[ends, :2]
            rebuilt = integrate(label, yaw[ends])
            rebuilt[:, :2] += lag_correction(label, yaw[ends]-yaw[ends][0], lag_gain())
            heading_frame.append(float(np.linalg.norm(stored[-1, :2]-truth[-1, :2])))
            world.append(float(np.max(np.linalg.norm(rebuilt[:, :2]-truth[:, :2], axis=1))))
            frames += len(t)
        report['splits'][split] = {'episodes': len(world), 'frames': frames,
                                   'heading_frame_label_endpoint_mean_m': float(np.mean(heading_frame)),
                                   'label_with_compensation_max_error_mean_m': float(np.mean(world)),
                                   'label_with_compensation_max_error_max_m': float(np.max(world))}
        print('labels', split, report['splits'][split], flush=True)
    write_json(REPORT/'label_audit.json', report)
    return report


def active_jobs():
    """PIDs of data, training or workflow processes started from this project."""
    running = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f'/proc/{entry}/cmdline', 'rb') as f:
                command = f.read()
            cwd = os.readlink(f'/proc/{entry}/cwd')
        except OSError:
            continue
        args = command.split(b'\0')
        if cwd == str(ROOT) and b'-m' in args and any(a in (b'g1_odom', b'g1_odom.workflow', b'g1_odom.dataset')
                                                      for a in args):
            if b'serve' not in args:
                running.append(int(entry))
    return running


def stop_jobs():
    stopped = []
    for pid in active_jobs():
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            stopped.append(pid)
        except OSError:
            pass
    progress(status='stopped')
    return {'ok': True, 'stopped': stopped}


def main():
    import fcntl
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('all', *STAGES), default='all')
    p.add_argument('--device')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--skip-online', action='store_true')
    a = p.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    selected = STAGES if a.stage == 'all' else (a.stage,)
    with open(RUN/'workflow.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('workflow already running')
        if 'collect' in selected:
            from .dataset import collect
            progress(status='collecting')
            collect(workers=a.workers)
        if 'audit' in selected:
            from .dataset import audit
            progress(status='auditing')
            write_json(REPORT/'data_audit.json', audit())
        if 'labels' in selected:
            label_audit()
        if 'train' in selected:
            from .train import train_all
            train_all(a.device)
        if 'calibrate' in selected:
            from .train import calibrate
            calibrate()
        if 'evaluate' in selected:
            from .evaluate import evaluate
            evaluate(skip_online=a.skip_online)


if __name__ == '__main__':
    main()
