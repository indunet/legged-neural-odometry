import numpy as np
import pytest
import torch

from g1_odom import trainer


def test_epoch_resume_matches_uninterrupted_training(tmp_path,monkeypatch):
    rng=np.random.default_rng(43)
    rows=[{'features':rng.normal(size=(85,25)).astype(np.float32),
           'label':rng.normal(size=(85,2)).astype(np.float32)*.1,
           'label_names':np.array(['vx','vy'])} for _ in range(6)]
    monkeypatch.setattr(trainer,'load_dataset',lambda _:rows)
    kwargs=dict(datasets=[tmp_path/'data'],features='noankle',epochs=2,steps_per_epoch=3,
                batch=16,seed=2,device='cpu',outputs=('vx','vy'))
    trainer.train(out_dir=tmp_path/'full',**kwargs)
    def interrupted(epoch,*_):
        if epoch==1:raise RuntimeError('simulated interruption')
    with pytest.raises(RuntimeError,match='simulated interruption'):
        trainer.train(out_dir=tmp_path/'resumed',progress=interrupted,**kwargs)
    trainer.train(out_dir=tmp_path/'resumed',**kwargs)
    a=torch.load(tmp_path/'full/checkpoint.pt',weights_only=False)
    b=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=False)
    for key in a['state_dict']:
        torch.testing.assert_close(a['state_dict'][key],b['state_dict'][key],rtol=0,atol=0)


def test_stationary_gate_keeps_slow_motion_and_releases_on_restart():
    from g1_odom.standstill import StandstillDetector
    detector=StandstillDetector()
    frame=np.zeros(25);frame[5]=9.81
    for _ in range(100):detector.update(frame)
    assert np.linalg.norm(detector.gate_velocity([.005,0.]))<1e-5
    np.testing.assert_array_equal(detector.gate_velocity([.03,0.]),[.03,0.])
    frame[17]=.2;detector.update(frame)
    np.testing.assert_array_equal(detector.gate_velocity([.005,0.]),[.005,0.])
