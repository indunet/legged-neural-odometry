import pytest

from g1_odom.common import ROOT
from g1_odom.dataset import configs,COUNTS,GENERATION


def test_trajectories_are_separate_and_cover_locked_counts():
    jobs=configs();assert len(jobs)==3110
    seeds={j['seed'] for j in jobs};assert len(seeds)==3110 and min(seeds)>=400000000
    for split,count in COUNTS.items():
        assert sum(j['split']==split for j in jobs)==count
    assert all(j['generation']==GENERATION and j['policy']=='flat' and j['course']=={'kind':'flat'} for j in jobs)
    assert all(len(j['physics']['kp_scale'])==len(j['physics']['kd_scale'])==29 for j in jobs)


def test_schedules_include_stationary_initialization_and_continuous_commands():
    jobs=configs()
    targets=[]
    for job in jobs:
        segments=job['schedule'];assert segments[0]['t0']==0 and segments[0]['t1']>=2.
        assert segments[0]['target']==[0.,0.,0.]
        assert all(a['t1']==b['t0'] for a,b in zip(segments,segments[1:]))
        targets.extend(s['target'][0] for s in segments)
    assert len(set(targets))>1000
    assert min(targets)<-.4 and max(targets)>1.


def test_training_loader_rejects_foreign_data_roots():
    from g1_odom.train import episode_loader
    for path in (ROOT/'data/other/train',ROOT/'releases'):
        with pytest.raises(ValueError,match='only the G1 planar'):
            episode_loader(path)
