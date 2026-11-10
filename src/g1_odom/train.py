"""Training: planar mean recipe on world-low-pass labels, frozen-mean
variance head, validation-only window/ensemble choice, ONNX export and calibration.

``trainer.py`` is reused as is; only its data loader is swapped for one that returns
world-low-pass labels.
"""
import json
from pathlib import Path
import time

import numpy as np

from .common import ROOT, FEATURE_NAMES, FRAME_DT, sha256, write_json
from .dataset import DATA, GENERATION, TRAIN_SPLITS, VAL_SPLITS
from .labels import LABEL_FRAME, LABEL_TAU, lag_gain, relabel
from .training_data import load_dataset

RUN = ROOT/'runs'
REPORT = ROOT/'reports'
RELEASE = ROOT/'releases'
SEEDS = (0, 1, 2)
WINDOWS = (25, 50)
WINDOW_GAIN_REQUIRED = .05
VARIANCE = {'epochs': 12, 'steps_per_epoch': 300, 'batch': 2048, 'lr': 1e-3, 'weight_decay': 1e-4,
            'loss': 'Gaussian NLL 0.5*(log var + r^2/var), mean frozen', 'selection': 'minimum in-pool validation NLL'}
PROGRESS = ROOT/'runs/progress.json'


def progress(**values):
    write_json(PROGRESS, {'generation': GENERATION, 'updated': time.time(), **values})


def episode_loader(path, trim_falls=True):
    if not Path(path).resolve().is_relative_to(DATA.resolve()):
        raise ValueError('only the G1 planar data root is permitted')
    return [relabel(row) for row in load_dataset(path, trim_falls)]


def training_data_sha256():
    """Hash of every training file, audited once per run directory."""
    from .dataset import audit
    path = RUN/'training_data.json'
    if not path.exists():
        write_json(path, audit(TRAIN_SPLITS))
    return json.loads(path.read_text())['files_sha256']


def lineage():
    return {'labels_sha256': sha256(ROOT/'src/g1_odom/labels.py'),
            'trainer_sha256': sha256(ROOT/'src/g1_odom/trainer.py'),
            'training_data_sha256': training_data_sha256()}


def member_dir(seed, window):
    return RUN/f'w{window}_seed{seed}'


def _checked(path, expected):
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError(f'{path} was produced from different labels, trainer or data; use a new run directory')
    return result


def train_mean(seed, window, device):
    """Stage 1: the planar recipe (TCN24, Huber 0.1, AdamW, OneCycle, 40x600)."""
    from . import trainer
    folder = member_dir(seed, window)
    folder.mkdir(parents=True, exist_ok=True)
    done = _checked(folder/'mean_complete.json', lineage())
    if done:
        return done

    def on_epoch(epoch, total, row):
        progress(status='training_mean', seed=seed, window=window, epoch=epoch, epochs=total, record=row)
        print('mean', window, seed, epoch, row, flush=True)
    original = trainer.load_dataset
    trainer.load_dataset = episode_loader
    try:
        metrics = trainer.train([DATA/s for s in TRAIN_SPLITS], folder, arch='tcn', features='noankle',
                                  window=window, epochs=40, steps_per_epoch=600, batch=2048, lr=.003,
                                  seed=seed, val_fraction=.1, init=None, outputs=('vx', 'vy'),
                                  static_boost=0., channels=24, select='loss', device=device, progress=on_epoch)
    finally:
        trainer.load_dataset = original
    result = {'seed': seed, 'window': window, 'metrics': metrics, 'label_frame': LABEL_FRAME, **lineage()}
    write_json(folder/'mean_complete.json', result)
    return result


def load_gaussian(folder, device='cpu', state='checkpoint'):
    import torch
    from .model import TCNGaussian
    cp = torch.load(Path(folder)/'checkpoint.pt', map_location='cpu', weights_only=False)
    model = TCNGaussian(len(cp['feature_index']), cp['mean'], cp['std'], channels=cp['channels'], n_out=2)
    missing, unexpected = model.load_state_dict(cp['state_dict'], strict=False)
    if unexpected or any(not k.startswith('var_head.') for k in missing):
        raise ValueError(f'checkpoint does not match the TCN mean path: {missing} {unexpected}')
    if state == 'gaussian':
        model.load_state_dict(torch.load(Path(folder)/'gaussian.pt', map_location='cpu', weights_only=False)['state_dict'])
    return model.to(device), cp


def _nll(model, data, ends, batch=4096):
    import torch
    total = 0.
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ends), batch):
            x, y = data.batch(ends[i:i+batch])
            mean, log_var = model(x)
            total += float((.5*(log_var+(y-mean)**2*torch.exp(-log_var))).sum())
    model.train()
    return total/(max(1, len(ends))*2)


def train_variance(seed, window, device):
    """Stage 2: mean path frozen; only the variance head learns, on the stage-1 split."""
    import torch
    from . import trainer
    folder = member_dir(seed, window)
    expected = {**lineage(), 'mean_checkpoint_sha256': sha256(folder/'checkpoint.pt')}
    done = _checked(folder/'variance_complete.json', expected)
    if done:
        return done
    torch.manual_seed(seed)
    model, cp = load_gaussian(folder, device)
    model.freeze_mean()
    episodes = [e for s in TRAIN_SPLITS for e in episode_loader(DATA/s)]
    train_eps, val_eps = trainer.split_episodes(episodes, .1, seed)
    train = trainer.Windows(train_eps, cp['feature_index'], window, device, (0, 1))
    val = trainer.Windows(val_eps, cp['feature_index'], window, device, (0, 1))
    rng = np.random.default_rng(seed+1000)
    val_ends = val.ends if len(val) <= 60000 else np.random.default_rng(0).choice(val.ends, 60000, replace=False)
    with torch.no_grad():
        x, y = train.batch(rng.choice(train.ends, 50000))
        residual = ((model(x)[0]-y)**2).mean(0)
        model.var_head[-1].weight.zero_()
        model.var_head[-1].bias.copy_(torch.log(torch.expm1(torch.clamp(residual-model.var_floor, min=1e-9))))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=VARIANCE['lr'], weight_decay=VARIANCE['weight_decay'])
    total = VARIANCE['epochs']*VARIANCE['steps_per_epoch']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total)
    reference = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.startswith('var_head.')}
    state_file = folder/'variance_state.pt'
    history, best, first = [], None, 0
    if state_file.exists():
        saved = torch.load(state_file, map_location=device, weights_only=False)
        if saved['expected'] != expected:
            raise ValueError('variance resume configuration differs')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        history, best, first = saved['history'], saved['best'], saved['epoch']+1
    for epoch in range(first, VARIANCE['epochs']):
        running = 0.
        for _ in range(VARIANCE['steps_per_epoch']):
            x, y = train.batch(rng.choice(train.ends, VARIANCE['batch']))
            mean, log_var = model(x)
            loss = (.5*(log_var+(y-mean)**2*torch.exp(-log_var))).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
        row = {'epoch': epoch+1, 'train_nll': running/VARIANCE['steps_per_epoch'], 'val_nll': _nll(model, val, val_ends)}
        history.append(row)
        if best is None or row['val_nll'] < best['val_nll']:
            best = {**row, 'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        progress(status='training_variance', seed=seed, window=window, epoch=epoch+1, epochs=VARIANCE['epochs'], record=row)
        print('variance', window, seed, row, flush=True)
        torch.save({'expected': expected, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'numpy_rng': rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state(), 'history': history, 'best': best}, folder/'variance_state.tmp.pt')
        (folder/'variance_state.tmp.pt').replace(state_file)
    for key, value in reference.items():
        if not torch.equal(best['state_dict'][key].to(value.device), value):
            raise RuntimeError(f'variance training changed the mean path: {key}')
    torch.save({'state_dict': best['state_dict'], 'window': window, 'feature_index': cp['feature_index']}, folder/'gaussian.pt')
    result = {'seed': seed, 'window': window, 'selected_epoch': best['epoch'], 'val_nll': best['val_nll'],
              'history': history, 'recipe': VARIANCE, **expected}
    write_json(folder/'variance_complete.json', result)
    return result


def predict(sess, features, ends, window):
    offsets = np.arange(-window+1, 1)
    chunks = [sess.run(None, {'features': features[ids[:, None]+offsets]})
              for ids in (ends[i:i+2048] for i in range(0, len(ends), 2048))]
    return [np.concatenate(parts) for parts in zip(*chunks)]


def first_frame(times, window=50):
    """Common scoring start: after calibration and a full window of either size."""
    return max(window-1, int(np.searchsorted(times, 1.5)))


def validation_episodes():
    return [e for s in VAL_SPLITS for e in episode_loader(DATA/s)]


def _mean_session(folder):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(Path(folder)/'estimator.onnx'), options, providers=['CPUExecutionProvider'])


def _validation_scores(predictions, episodes):
    from .evaluate import integrate, relative_pose
    from .labels import lag_correction
    sq, n, endpoints = np.zeros(2), 0, []
    for velocity, row in zip(predictions, episodes):
        ends = np.arange(first_frame(row['time']), len(row['time']))
        residual = velocity-row['label'][ends, :2]
        sq += (residual**2).sum(0)
        n += len(residual)
        yaw = np.unwrap(row['pose'][ends, 3])
        estimate = integrate(velocity, yaw)
        estimate[:, :2] += lag_correction(velocity, yaw-yaw[0], lag_gain())
        truth = relative_pose(np.c_[row['pose'][ends, :3], yaw])
        endpoints.append(float(np.linalg.norm(estimate[-1, :2]-truth[-1, :2])))
    rmse = np.sqrt(sq/n)
    return {'velocity_rmse': rmse.tolist(), 'planar_rmse_sum': float(rmse.sum()),
            'endpoint_mean_m': float(np.mean(endpoints)), 'episodes': len(episodes)}


def choose_window(episodes):
    """Seed 0 at 25 and 50 frames, scored on the extra validation set only."""
    scores = {}
    for window in WINDOWS:
        sess = _mean_session(member_dir(0, window))
        predictions = [predict(sess, e['features'], np.arange(first_frame(e['time']), len(e['time'])), window)[0]
                       for e in episodes]
        scores[window] = _validation_scores(predictions, episodes)
    gain = 1.-scores[50]['endpoint_mean_m']/scores[25]['endpoint_mean_m']
    chosen = 50 if gain >= WINDOW_GAIN_REQUIRED else 25
    result = {'rule': f'adopt 50 frames only if extra-validation mean endpoint error improves by >= {WINDOW_GAIN_REQUIRED:.0%}',
              'scores': {str(k): v for k, v in scores.items()}, 'endpoint_gain': gain, 'chosen_window': chosen,
              'data': 'extra validation val+stop_val; world-low-pass labels; ideal yaw; lag compensation; no stationary gate'}
    write_json(REPORT/'window_selection.json', result)
    return result


def export_member(folder, window, metadata):
    import torch
    from .model import parameter_count
    model, _ = load_gaussian(folder, state='gaussian')
    model = model.eval().cpu()
    metadata = {**metadata, 'parameters': parameter_count(model)}
    path = Path(folder)/'estimator_gaussian.onnx'
    torch.onnx.export(model, torch.zeros(1, window, len(FEATURE_NAMES)), str(path), input_names=['features'],
                      output_names=['velocity', 'log_var'],
                      dynamic_axes={'features': {0: 'batch'}, 'velocity': {0: 'batch'}, 'log_var': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    _set_metadata(path, {**metadata, 'uncertainty': 'gaussian_log_var'})
    return path


def _set_metadata(path, metadata):
    import onnx
    proto = onnx.load(str(path))
    old = {p.key: p.value for p in proto.metadata_props}
    old.update({k: str(v) for k, v in metadata.items()})
    del proto.metadata_props[:]
    for key, value in old.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, str(path))


def base_metadata(window, parameters):
    return {'feature_names': ','.join(FEATURE_NAMES), 'window': window, 'frame_rate': 50.,
            'joint_sample_rate_hz': 50., 'encoder_interface_rate_hz': 500, 'label_names': 'vx,vy',
            'label_tau': LABEL_TAU, 'label_frame': LABEL_FRAME, 'lag_compensation_s': lag_gain(),
            'output_frame': 'heading (gravity aligned, pelvis yaw)', 'arch': 'tcn_gauss', 'parameters': parameters,
            'robot': 'Unitree G1 29-DOF', 'training_recipe': 'planar mean + frozen-mean Gaussian NLL variance',
            'stationary_gate': 'standstill', 'heading_source': 'imu_yaw', 'data_generation': GENERATION, 'init': 'from_scratch', **lineage()}


def export_ensemble(window, primary, path):
    """One ONNX graph; outputs velocity, aleatoric_var, epistemic_var (per axis)."""
    import torch
    from .model import GaussianEnsemble, parameter_count
    from .simulation import session
    members = [load_gaussian(member_dir(seed, window), state='gaussian')[0].eval().cpu() for seed in SEEDS]
    ensemble = GaussianEnsemble(members, primary).eval()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(ensemble, torch.zeros(1, window, len(FEATURE_NAMES)), str(path), input_names=['features'],
                      output_names=['velocity', 'aleatoric_var', 'epistemic_var'],
                      dynamic_axes={n: {0: 'batch'} for n in ('features', 'velocity', 'aleatoric_var', 'epistemic_var')},
                      opset_version=17, dynamo=False)
    parameters = parameter_count(ensemble)
    _set_metadata(path, {**base_metadata(window, parameters), 'uncertainty': 'ensemble_gaussian',
                         'members': len(members), 'seeds': ','.join(map(str, SEEDS)),
                         'mean_source': 'ensemble average' if primary is None else f'seed {SEEDS[primary]}'})
    cp = members[0]
    x = (np.random.default_rng(177).normal(size=(256, window, len(FEATURE_NAMES)))*cp.norm.std.numpy()
         + cp.norm.mean.numpy()).astype(np.float32)
    with torch.no_grad():
        expected = [t.numpy() for t in ensemble(torch.from_numpy(x))]
    actual = session(path).run(None, {'features': x})
    error = max(float(np.max(np.abs(a-e)/(np.abs(e)+1e-6 if i else 1.))) for i, (a, e) in enumerate(zip(actual, expected)))
    if error > 1e-4:
        raise RuntimeError(f'ONNX ensemble mismatch {error}')
    return {'path': str(path), 'parameters': parameters, 'max_parity_error': error}


def choose_mean_source(window, episodes):
    """Ensemble average unless one seed alone has lower extra-validation planar RMSE."""
    sessions = [_mean_session(member_dir(seed, window)) for seed in SEEDS]
    per_member = []
    for sess in sessions:
        per_member.append([predict(sess, e['features'], np.arange(first_frame(e['time']), len(e['time'])), window)[0]
                           for e in episodes])
    scores = {f'seed{seed}': _validation_scores(p, episodes) for seed, p in zip(SEEDS, per_member)}
    average = [np.mean(group, axis=0) for group in zip(*per_member)]
    scores['ensemble'] = _validation_scores(average, episodes)
    best = min(scores, key=lambda k: scores[k]['planar_rmse_sum'])
    primary = None if best == 'ensemble' else SEEDS.index(int(best[4:]))
    return primary, {'rule': 'lowest extra-validation vx+vy RMSE among the ensemble average and each seed',
                     'scores': scores, 'chosen': best}


def train_all(device=None):
    from .common import gpu_device
    device = device or gpu_device()
    RUN.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    train_mean(0, 25, device)
    train_mean(0, 50, device)
    episodes = validation_episodes()
    selection = choose_window(episodes)
    window = selection['chosen_window']
    stages = {}
    for seed in SEEDS:
        stages[seed] = {'mean': train_mean(seed, window, device)}
    for seed in SEEDS:
        stages[seed]['variance'] = train_variance(seed, window, device)
    primary, mean_choice = choose_mean_source(window, episodes)
    staging = RUN/f'w{window}_ensemble'/'estimator.onnx'
    exported = export_ensemble(window, primary, staging)
    for seed in SEEDS:
        export_member(member_dir(seed, window), window, base_metadata(window, None))
    result = {'generation': GENERATION, 'window_selection': selection, 'window': window,
              'mean_source': mean_choice, 'primary': primary, 'export': exported,
              'staging_sha256': sha256(staging), 'members': {str(s): stages[s] for s in SEEDS}, **lineage()}
    write_json(REPORT/'training.json', result)
    progress(status='trained', window=window)
    return result


# --------------------------------------------------------------------------- calibration

def _member_outputs(sess, episodes, window):
    rows = []
    for e in episodes:
        ends = np.arange(first_frame(e['time']), len(e['time']))
        rows.append((ends, *predict(sess, e['features'], ends, window)))
    return rows


def _gate(e, ends, velocity):
    from .standstill import StandstillDetector
    from .uncertainty import gate_keep
    degrees = StandstillDetector().episode(e['features'])[ends]
    return gate_keep(velocity, degrees)


def fit_velocity_scales(residual, aleatoric, epistemic, keep):
    from scipy.optimize import minimize
    static = keep < .5
    static_var = (residual[static]**2).mean(0) if static.sum() > 100 else np.full(2, 1e-6)
    static_var = np.maximum(static_var, 1e-8)
    k2 = keep[:, None]**2
    floor = 1e-8

    def nll(log_scales):
        sa, se = np.exp(log_scales)
        var = k2*(sa**2*aleatoric+se**2*epistemic)+(1-k2)*static_var+floor
        return float(np.mean(.5*(np.log(var)+residual**2/var)))
    fit = minimize(nll, np.zeros(2), method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-8, 'maxiter': 2000})
    sa, se = np.exp(fit.x)
    return {'aleatoric_scale': float(sa), 'epistemic_scale': float(se), 'static_var': static_var.tolist(),
            'static_frames': int(static.sum()), 'var_floor': floor, 'velocity_nll': float(fit.fun)}


def fit_temporal(normalized, max_lag=100):
    """Integral correlation time of the demeaned residual and the per-run bias share."""
    num, den = np.zeros(max_lag+1), 0.
    for z in normalized:
        d = z-z.mean(0)
        den += float((d**2).sum())
        for k in range(max_lag+1):
            if len(d) > k:
                num[k] += float((d[:len(d)-k]*d[k:]).sum())
    acf = num/den
    cut = int(np.argmax(acf < .02)) if (acf < .02).any() else max_lag+1
    tc = FRAME_DT*(.5+acf[1:cut].sum())
    means = np.array([z.mean(0) for z in normalized])
    durations = np.array([len(z)*FRAME_DT for z in normalized])[:, None]
    white = 2*tc/durations
    rho = float(np.clip(np.mean(means**2-white)/np.mean(1-white), .01, .9))
    return {'correlation_time_s': float(tc), 'bias_fraction': rho, 'bias_time_s': None,
            'autocorrelation': acf[:min(len(acf), 51)].tolist()}


def smooth_quality(x, tau=1.):
    """Offline equivalent of ``QualityMonitor``: EMA starting at the first value."""
    from scipy.signal import lfilter
    alpha = FRAME_DT/(tau+FRAME_DT)
    return lfilter([alpha], [1., alpha-1.], x, zi=[(1.-alpha)*x[0]])[0]


def calibrate(staging=None):
    """Fit the uncertainty model on the extra validation set (never the test set)."""
    from .simulation import session
    from .uncertainty import CHI2_2DOF_95, epistemic_std
    from .evaluate import replay
    training = json.loads((REPORT/'training.json').read_text())
    window = training['window']
    staging = Path(staging or RUN/f'w{window}_ensemble'/'estimator.onnx')
    if sha256(staging) != training['staging_sha256']:
        raise RuntimeError('staged ensemble differs from the trained selection')
    progress(status='calibrating', window=window)
    episodes = validation_episodes()
    sess = session(staging)
    outputs = _member_outputs(sess, episodes, window)
    residual, alea, epi, keep, per_episode = [], [], [], [], []
    for e, (ends, velocity, a, s) in zip(episodes, outputs):
        k = _gate(e, ends, velocity)
        r = e['label'][ends, :2]-velocity*k[:, None]
        residual.append(r); alea.append(a); epi.append(s); keep.append(k); per_episode.append(len(r))
    R, A, E, K = (np.concatenate(x) for x in (residual, alea, epi, keep))
    cal = fit_velocity_scales(R, A, E, K)
    from .uncertainty import velocity_variance
    normalized = [r/np.sqrt(velocity_variance(a, s, k, cal)) for r, a, s, k in zip(residual, alea, epi, keep)]
    cal.update(fit_temporal(normalized))
    cal['pose_scale'] = 1.
    nees = []
    for e, out in zip(episodes, outputs):
        replayed = replay(e, out, window, cal, lag=True, yaw_drift_deg_min=0.)
        error, cov = replayed['xy_error'], replayed['pose_cov']
        for i in range(50, len(error), 50):
            P = cov[i][:2, :2]
            nees.append(float(error[i]@np.linalg.solve(P, error[i])))
    nees = np.array(nees)
    cal['pose_scale_unscaled_nees_mean'] = float(nees.mean())
    cal['pose_scale'] = float(np.quantile(nees, .95)/CHI2_2DOF_95)
    smoothed = np.concatenate([smooth_quality(epistemic_std(s, cal)) for s in epi])
    cal.update({'quality_tau_s': 1., 'quality_warn': float(np.quantile(smoothed, .99)),
                'quality_alarm': float(np.quantile(smoothed, .999)),
                'quality_rule': 'EMA(1 s) of calibrated seed-disagreement speed std vs extra-validation p99 / p99.9'})
    z = np.concatenate(normalized)
    cal['validation'] = {'episodes': len(episodes), 'frames': int(len(R)),
                         'coverage_1sigma': float(np.mean(np.abs(z) < 1)), 'coverage_2sigma': float(np.mean(np.abs(z) < 2)),
                         'pose_nees_samples': int(len(nees))}
    cal.update({'data': 'extra validation val+stop_val only; ideal simulated yaw; yaw drift excluded while fitting',
                'yaw_drift_deg_min_default': 1., 'window': window, 'staging_sha256': training['staging_sha256']})
    RELEASE.mkdir(parents=True, exist_ok=True)
    release = RELEASE/'estimator.onnx'
    release.write_bytes(staging.read_bytes())
    _set_metadata(release, {'calibration': json.dumps(cal, separators=(',', ':'))})
    write_json(RELEASE/'calibration.json', cal)
    members = RELEASE/'members'
    members.mkdir(exist_ok=True)
    for seed in SEEDS:
        (members/f'estimator_seed{seed}.onnx').write_bytes((member_dir(seed, window)/'estimator_gaussian.onnx').read_bytes())
    model = {'generation': GENERATION, 'onnx_sha256': sha256(release), 'window': window,
             'mean_source': training['mean_source']['chosen'], 'window_selection': training['window_selection'],
             'rule': 'window, mean source and calibration from extra validation only; tests run after locking',
             **lineage()}
    write_json(RELEASE/'model.json', model)
    write_json(RELEASE/'runtime.json', {
        'heading_source': 'imu_yaw', 'stationary_gate': 'standstill', 'label_frame': LABEL_FRAME,
        'lag_compensation_s': lag_gain(), 'window': window, 'network_sha256': model['onnx_sha256'],
        'outputs': ['velocity', 'velocity_cov', 'pose', 'pose_cov', 'uncertainty', 'quality'],
        'pose_cov_order': 'x, y, yaw; row-major 3x3', 'velocity_cov_order': 'vx, vy; row-major 2x2',
        'yaw_drift_deg_min_default': 1., 'simulation_heading': 'ideal simulated IMU attitude; not hardware accuracy'})
    write_json(REPORT/'calibration.json', cal)
    progress(status='calibrated', model_sha256=model['onnx_sha256'])
    return cal
