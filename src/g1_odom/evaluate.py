"""Locked-weight evaluation of the released odometry: accuracy and uncertainty."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp

import numpy as np

from .common import FRAME_DT, sha256, write_json
from .dataset import DATA, TEST_SPLITS
from .labels import lag_correction, lag_gain
from .uncertainty import CHI2_2DOF_95, PoseCovariance, epistemic_std, velocity_variance
from .train import RELEASE, REPORT, episode_loader, first_frame, predict, progress, smooth_quality, _gate

MARKS_M = (5, 10, 20)


def integrate(velocities, heading, dt=FRAME_DT):
    """Trapezoidal planar integration; rows [x, y, z, yaw] relative to the first heading."""
    pose = np.zeros((len(velocities), 4))
    angles = np.unwrap(heading)-heading[0]
    pose[:, 3] = angles
    v = np.zeros((len(velocities), 3))
    v[:, :velocities.shape[1]] = velocities
    average = .5*(v[1:]+v[:-1])
    mid = .5*(angles[1:]+angles[:-1])
    c, s = np.cos(mid), np.sin(mid)
    increments = np.c_[c*average[:, 0]-s*average[:, 1], s*average[:, 0]+c*average[:, 1], average[:, 2]]*dt
    pose[1:, :3] = np.cumsum(increments, axis=0)
    return pose


def relative_pose(pose):
    """Rows [x, y, z, yaw] expressed relative to the first row's position and heading."""
    x = pose.copy()
    x[:, :3] -= x[0, :3]
    c, s = np.cos(pose[0, 3]), np.sin(pose[0, 3])
    x[:, :2] = np.c_[c*x[:, 0]+s*x[:, 1], -s*x[:, 0]+c*x[:, 1]]
    x[:, 3] = np.unwrap(pose[:, 3])-pose[0, 3]
    return x


def aggregate(rows):
    valid = [r for r in rows if r.get('valid')]
    out = {'episodes': len(rows), 'valid': len(valid), 'fell': sum(r.get('fell', False) for r in valid)}
    for key in ('ate_xy_m', 'endpoint_xy_m'):
        values = [r[key] for r in valid if r.get(key) is not None]
        if values:
            out[key] = {'mean': float(np.mean(values)), 'median': float(np.median(values)),
                        'p95': float(np.percentile(values, 95))}
    if valid:
        out['velocity_rmse_episode_mean'] = np.mean([r['velocity_rmse'] for r in valid], axis=0).tolist()
    distance = {}
    for meters in map(str, MARKS_M):
        entries = [r.get('rpe', {}).get(meters, {}) for r in valid]
        entries = [r for r in entries if r.get('segments') and r.get('position_mean_m') is not None]
        count = sum(r['segments'] for r in entries)
        distance[meters] = {'segments': count, 'position_mean_m':
                            sum(r['segments']*r['position_mean_m'] for r in entries)/count if count else None}
    out['rpe'] = distance
    return out


def model_window(sess):
    return int(sess.get_inputs()[0].shape[1])


def load_calibration(sess):
    raw = sess.get_modelmeta().custom_metadata_map.get('calibration')
    return json.loads(raw) if raw else None


def replay(row, out, window, cal=None, lag=False, yaw_drift_deg_min=0., heading_drift_deg_min=0.):
    """Offline twin of ``Odometry.update`` on recorded features with ideal yaw.

    ``out`` is (ends, velocity[, aleatoric_var, epistemic_var]).
    """
    ends, raw = out[0], out[1]
    keep = _gate(row, ends, raw)
    velocity = raw*keep[:, None]
    times = row['time'][ends]
    yaw = np.unwrap(row['pose'][:, 3])[ends]
    heading = yaw+np.deg2rad(heading_drift_deg_min)*(times-times[0])/60.
    estimate = integrate(velocity, heading)
    if lag:
        estimate[:, :2] += lag_correction(velocity, heading-heading[0], lag_gain())
    truth = relative_pose(np.c_[row['pose'][ends, :3], yaw])
    result = {'ends': ends, 'raw': raw, 'velocity': velocity, 'keep': keep, 'estimate': estimate, 'truth': truth,
              'xy_error': estimate[:, :2]-truth[:, :2]}
    if cal is not None and len(out) >= 4:
        variance = velocity_variance(out[2], out[3], keep, cal)
        tracker = PoseCovariance(cal, yaw_drift_deg_min)
        angles = heading-heading[0]
        covs = np.zeros((len(ends), 3, 3))
        covs[0] = tracker.pose_cov
        std = np.sqrt(variance)
        for k in range(1, len(ends)):
            tracker.step(FRAME_DT, .5*(angles[k]+angles[k-1]), .5*(velocity[k]+velocity[k-1]), std[k])
            covs[k] = tracker.P[:3, :3]
        result.update(variance=variance, pose_cov=covs, epistemic_speed=epistemic_std(out[3], cal))
    return result


def _rpe(estimate, truth, distance):
    rpe = {}
    for meters in MARKS_M:
        position = []
        for first in range(0, len(distance), 50):
            last = np.searchsorted(distance, distance[first]+meters)
            if last >= len(distance):
                continue

            def delta(track):
                v = track[last, :2]-track[first, :2]
                c, s = np.cos(track[first, 3]), np.sin(track[first, 3])
                return [c*v[0]+s*v[1], -s*v[0]+c*v[1]]
            position.append(float(np.linalg.norm(np.array(delta(estimate))-delta(truth))))
        rpe[str(meters)] = {'segments': len(position), 'position_mean_m': float(np.mean(position)) if position else None,
                            'yaw_mean_deg': 0. if position else None}
    return rpe


def score(row, replayed):
    ends = replayed['ends']
    label = row['label'][ends, :2]
    xy = np.linalg.norm(replayed['xy_error'], axis=1)
    distance = np.cumsum(np.linalg.norm(label, axis=1))*FRAME_DT
    distance -= distance[0]
    metrics = {'valid': True, 'file': row['info']['file'], 'frames': len(ends),
               'fell': bool(row['info'].get('fell', False)), 'finished': not row['info'].get('fell', False),
               'endpoint_xy_m': float(xy[-1]), 'ate_xy_m': float(np.sqrt(np.mean(xy**2))),
               'velocity_rmse': np.sqrt(np.mean((replayed['velocity']-label)**2, axis=0)).tolist(),
               'path_m': float(distance[-1]), 'rpe': _rpe(replayed['estimate'], replayed['truth'], distance)}
    if 'pose_cov' not in replayed:
        return metrics, None
    residual = label-replayed['velocity']
    variance = replayed['variance']
    cov = replayed['pose_cov']
    error = replayed['xy_error']

    def nees(i):
        return float(error[i]@np.linalg.solve(cov[i][:2, :2], error[i]))
    samples = [nees(i) for i in range(50, len(ends), 50)]
    marks = {}
    for meters in MARKS_M:
        index = int(np.searchsorted(distance, meters))
        marks[str(meters)] = nees(index) if index < len(ends) else None
    marks['endpoint'] = nees(len(ends)-1)
    metrics.update(endpoint_sigma_xy_m=float(np.sqrt(np.trace(cov[-1][:2, :2]))),
                   epistemic_speed_mean=float(np.mean(replayed['epistemic_speed'])),
                   velocity_rmse_norm=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))))
    pooled = {'residual': residual, 'variance': variance, 'nees': np.array(samples), 'marks': marks,
              'quality': smooth_quality(replayed['epistemic_speed'])}
    return metrics, pooled


def _ence(residual, variance, bins=10):
    values = []
    for axis in range(residual.shape[1]):
        order = np.argsort(variance[:, axis])
        for chunk in np.array_split(order, bins):
            rmv = np.sqrt(variance[chunk, axis].mean())
            values.append(abs(rmv-np.sqrt((residual[chunk, axis]**2).mean()))/rmv)
    return float(np.mean(values))


def uncertainty_summary(pooled, cal):
    from scipy.stats import spearmanr
    residual = np.concatenate([p['residual'] for p in pooled])
    variance = np.concatenate([p['variance'] for p in pooled])
    z = residual/np.sqrt(variance)
    nees = np.concatenate([p['nees'] for p in pooled])
    sub = np.random.default_rng(0).choice(len(z), min(len(z), 200000), replace=False)
    rank = spearmanr(np.sqrt(variance[sub]).ravel(), np.abs(residual[sub]).ravel()).statistic
    marks = {}
    for key in [*map(str, MARKS_M), 'endpoint']:
        values = np.array([p['marks'][key] for p in pooled if p['marks'].get(key) is not None])
        marks[key] = {'samples': int(len(values)),
                      'coverage_2sigma': float(np.mean(values < CHI2_2DOF_95)) if len(values) else None,
                      'nees_mean': float(values.mean()) if len(values) else None}
    quality = np.concatenate([p['quality'] for p in pooled])
    return {'frames': int(len(z)),
            'velocity_nll': float(np.mean(.5*(np.log(2*np.pi*variance)+z**2))),
            'coverage_1sigma': float(np.mean(np.abs(z) < 1)), 'coverage_2sigma': float(np.mean(np.abs(z) < 2)),
            'ence': _ence(residual, variance), 'spearman_sigma_abs_error': float(rank),
            'pose_nees_mean': float(nees.mean()) if len(nees) else None,
            'pose_coverage_2sigma': float(np.mean(nees < CHI2_2DOF_95)) if len(nees) else None,
            'pose_marks': marks,
            'quality_fraction': {'warn_or_alarm': float(np.mean(quality > cal['quality_warn'])),
                                 'alarm': float(np.mean(quality > cal['quality_alarm']))}}


def auroc(positive, negative):
    from scipy.stats import mannwhitneyu
    if not len(positive) or not len(negative):
        return None
    return float(mannwhitneyu(positive, negative).statistic/(len(positive)*len(negative)))


def evaluate_split(sess, episodes, cal, variants):
    """Every variant (replay keyword set) of the locked model on one split."""
    window = model_window(sess)
    results = {name: [] for name in variants}
    pooled = {name: [] for name in variants}
    for row in episodes:
        if len(row['time']) <= first_frame(row['time'])+2:
            continue
        ends = np.arange(first_frame(row['time']), len(row['time']))
        out = (ends, *predict(sess, row['features'], ends, window))
        for name, kwargs in variants.items():
            replayed = replay(row, out, window, cal if kwargs.get('lag') else None, **kwargs)
            metrics, uncertainty = score(row, replayed)
            results[name].append(metrics)
            if uncertainty is not None:
                pooled[name].append(uncertainty)
    return results, pooled


def run_online(name, command, seed, push, friction):
    from .runtime import Odometry
    from .simulation import Simulation
    sim = Simulation(seed=seed, noise=True)
    if friction:
        sim.model.geom_friction[:, 0] = friction
    odom = Odometry(RELEASE/'estimator.onnx')
    poses, covs, quality, truth = [], [], [], []
    for _ in range(45000):
        t = sim.data.time
        cmd = np.array(command if t > 2 else [0, 0, 0], dtype=float)
        if name == 'stop_go' and (t % 20) > 10:
            cmd[:] = 0
        sim.step(cmd, [50, 0, 0] if push and 10 < t < 10.2 else None)
        result = odom.update(sim.observe(imu_yaw=True))
        if result is not None:
            poses.append(result['pose'])
            covs.append(np.array(result['pose_cov']).reshape(3, 3))
            quality.append(result['quality'])
            truth.append(sim.truth()[1])
        if sim.fallen():
            break
    actual = relative_pose(np.array(truth))
    error = np.array(poses)[:, :2]-actual[:, :2]
    norm = np.linalg.norm(error, axis=1)
    P = covs[-1][:2, :2]
    return name, {'seconds': float(sim.data.time), 'fell': sim.fallen(),
                  'endpoint_xy_m': float(norm[-1]), 'ate_xy_m': float(np.sqrt(np.mean(norm**2))),
                  'endpoint_sigma_xy_m': float(np.sqrt(np.trace(P))),
                  'endpoint_nees': float(error[-1]@np.linalg.solve(P, error[-1])),
                  'quality': {q: float(np.mean(np.array(quality) == q)) for q in ('ok', 'warn', 'alarm')}}


def online(seed_offset=590000000):
    scenarios = {'standing': ([0, 0, 0], False, None), 'forward': ([.4, 0, 0], False, None),
                 'backward': ([-.25, 0, 0], False, None), 'lateral': ([0, .2, 0], False, None),
                 'circle': ([.3, 0, .35], False, None), 'stop_go': ([.4, 0, 0], False, None),
                 'low_friction': ([.4, 0, 0], False, .2), 'push': ([.4, 0, 0], True, None)}
    rows = {}
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(run_online, name, cmd, seed_offset+i, push, friction)
                   for i, (name, (cmd, push, friction)) in enumerate(scenarios.items())]
        for future in as_completed(futures):
            name, row = future.result()
            rows[name] = row
            print('online', name, row['endpoint_xy_m'], flush=True)
    return {'reference': 'ideal simulated IMU yaw; not hardware accuracy; pose_cov includes default 1 deg/min yaw drift',
            'scenarios': {n: rows[n] for n in scenarios}}


def evaluate(skip_online=False):
    from .simulation import session
    path = RELEASE/'estimator.onnx'
    model = json.loads((RELEASE/'model.json').read_text())
    if sha256(path) != model['onnx_sha256']:
        raise RuntimeError('release changed after calibration')
    sess = session(path)
    cal = load_calibration(sess)
    progress(status='evaluating')
    variants = {'no_lag_compensation': {}, 'released': {'lag': True, 'yaw_drift_deg_min': 0.}}
    result = {'locked_sha256': model['onnx_sha256'], 'calibration': cal,
              'heading_reference': 'ideal simulated IMU yaw; not real Unitree accuracy',
              'scoring': 'frames from 1.5 s; falls kept; position against true pelvis pose with initial alignment only',
              'datasets': {}}
    episodes_by_split = {}
    for split in TEST_SPLITS:
        rows, pooled = evaluate_split(sess, episode_loader(DATA/split, trim_falls=False), cal, variants)
        entry = {name: {'overall': aggregate(metrics), 'complete': aggregate([m for m in metrics if not m['fell']]),
                        'episodes': metrics} for name, metrics in rows.items()}
        entry['uncertainty'] = uncertainty_summary(pooled['released'], cal)
        complete = [p for p, m in zip(pooled['released'], rows['released']) if not m['fell']]
        entry['uncertainty_complete'] = uncertainty_summary(complete, cal) if complete else None
        episodes_by_split[split] = rows['released']
        result['datasets'][split] = entry
        print(split, round(100*entry['released']['overall']['endpoint_xy_m']['mean'], 2),
              entry['uncertainty']['coverage_1sigma'], flush=True)
        write_json(REPORT/'evaluation.json', result)
    ood = [m['epistemic_speed_mean'] for m in episodes_by_split['test_ood']]
    iid = [m['epistemic_speed_mean'] for m in episodes_by_split['test_iid']]
    every = [m for s in TEST_SPLITS for m in episodes_by_split[s]]
    threshold = float(np.quantile([m['velocity_rmse_norm'] for m in episodes_by_split['test_iid']], .9))
    high = [m['epistemic_speed_mean'] for m in every if m['velocity_rmse_norm'] > threshold]
    low = [m['epistemic_speed_mean'] for m in every if m['velocity_rmse_norm'] <= threshold]
    result['detection'] = {'auroc_ood_vs_iid': auroc(ood, iid), 'auroc_high_error_episodes': auroc(high, low),
                           'high_error_threshold_rmse': threshold, 'high_error_episodes': len(high),
                           'score': 'episode mean of calibrated seed-disagreement speed std'}
    sensitivity = {}
    episodes = episode_loader(DATA/'odometry', trim_falls=False)
    for drift in (1., 3.):
        rows, pooled = evaluate_split(sess, episodes, cal, {'released': {
            'lag': True, 'yaw_drift_deg_min': drift, 'heading_drift_deg_min': drift}})
        sensitivity[str(drift)] = {'endpoint_mean_m': aggregate(rows['released'])['endpoint_xy_m']['mean'],
                                   'uncertainty': uncertainty_summary(pooled['released'], cal)}
    result['yaw_drift_sensitivity'] = sensitivity
    if not skip_online:
        result['online'] = online()
    if sha256(path) != model['onnx_sha256']:
        raise RuntimeError('model changed during evaluation')
    write_json(REPORT/'evaluation.json', result)
    make_report(result)
    progress(status='complete', model_sha256=model['onnx_sha256'])
    return result


def _cm(value):
    return '—' if value is None else f'{100*value:.2f}'


def _pct(value):
    return '—' if value is None else f'{100*value:.1f}%'


def make_report(result):
    training = json.loads((REPORT/'training.json').read_text())
    label = json.loads((REPORT/'label_audit.json').read_text())
    cal = result['calibration']
    window = training['window_selection']
    floor = max(s['label_with_compensation_max_error_max_m'] for s in label['splits'].values())
    lines = ['# G1 二维里程计报告', '',
             '输入原始 IMU 与八个髋膝关节编码器（25 帧 × 25 维，50 Hz），输出骨盆航向坐标系的 vx、vy 及其不确定度；'
             '航向直接读取 IMU yaw，积分得到 x、y、yaw 与位姿协方差。',
             '全部数据由 G1 29 自由度仿真生成；以下使用理想仿真 IMU yaw，不代表宇树真机精度。', '',
             '## 方法', '',
             '- 标签：由记录的骨盆位置差分，在世界坐标系做 0.2 s 一阶低通，再旋转到当前航向。'
             '积分端加 (τ + dt/2)·(v_now − v_start)（世界系）抵消滤波滞后；'
             f"把标签当作网络输出积分，训练与验证集上整条轨迹最大位置误差不超过 {100*floor:.3f} cm。",
             '- 均值网络：TCN24（约 6k 参数），二维配方：Huber(0.1)、AdamW + OneCycleLR、batch 2048、40×600 steps、'
             '整轨迹 90/10 划分、按验证 Huber 选模；种子 0/1/2 从零训练。',
             '- 不确定度：冻结均值后，每个种子训练方差头（高斯负对数似然）；三种子合成一张图，输出均值、平均方差（偶然）'
             '与种子间方差（模型）。在额外验证集上校准后，用误差状态 [px, py, yaw, bx, by] 传播位姿协方差。', '',
             '## 选择与校准（只用验证集）', '',
             f"- 窗口：种子 0 在额外验证集上，25 帧终点均值 {_cm(window['scores']['25']['endpoint_mean_m'])} cm，"
             f"50 帧 {_cm(window['scores']['50']['endpoint_mean_m'])} cm；提升 ≥ 5% 才换，选定 **{training['window']} 帧**。",
             f"- 均值来源：{training['mean_source']['chosen']}（额外验证集 vx+vy RMSE 最低者）。",
             f"- 校准：偶然方差系数 {cal['aleatoric_scale']:.3f}，种子间方差系数 {cal['epistemic_scale']:.3f}，"
             f"相关时间 T_c = {cal['correlation_time_s']:.3f} s，偏置占比 ρ = {cal['bias_fraction']:.3f}，"
             f"位姿缩放 {cal['pose_scale']:.3f}；验证集速度 1σ／2σ 覆盖率 {_pct(cal['validation']['coverage_1sigma'])}／"
             f"{_pct(cal['validation']['coverage_2sigma'])}。", '',
             '## 精度（锁定权重后的独立测试）', '',
             '| 测试集 | 轨迹／跌倒 | 终点均值 (cm) | 终点 P95 (cm) | XY ATE (cm) | 5 m RPE (cm) | 不加滞后补偿的终点均值 (cm) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for split in TEST_SPLITS:
        d = result['datasets'][split]
        r, n = d['released']['overall'], d['no_lag_compensation']['overall']
        lines.append(f"| {split} | {r['episodes']} / {r['fell']} | {_cm(r['endpoint_xy_m']['mean'])} | {_cm(r['endpoint_xy_m']['p95'])} | "
                     f"{_cm(r['ate_xy_m']['mean'])} | {_cm(r['rpe']['5']['position_mean_m'])} | {_cm(n['endpoint_xy_m']['mean'])} |")
    ood = result['datasets']['test_ood']
    lines += ['', f"test_ood 为低摩擦、重负载、低刚度与扭矩折减的组合，{ood['released']['overall']['fell']} 条跌倒；"
              f"只看未跌倒轨迹，终点均值 {_cm(ood['released']['complete']['endpoint_xy_m']['mean'])} cm。"
              '跌倒时速度很大，滞后补偿会把跌倒瞬间的速度加到位姿上；部署时应在跌倒检测后停止或重置里程计。', '',
              f"静止门控在额外验证集上生效的帧数为 {cal.get('static_frames', '—')}（G1 平地策略在零速指令下站定，与 Unitree 部署一致，"
              '关节速度低于门限）；静止时的精度主要来自网络本身。', '',
              '## 不确定度（校准参数已锁定）', '',
              '| 测试集 | 速度 NLL | 1σ 覆盖 | 2σ 覆盖 | ENCE | σ 与 |误差| 秩相关 | 位姿 2σ 覆盖 | 位姿 NEES 均值 | 终点 2σ 覆盖 | warn+alarm 帧 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in TEST_SPLITS:
        u = result['datasets'][split]['uncertainty']
        lines.append(f"| {split} | {u['velocity_nll']:.2f} | {_pct(u['coverage_1sigma'])} | {_pct(u['coverage_2sigma'])} | "
                     f"{u['ence']:.3f} | {u['spearman_sigma_abs_error']:.2f} | {_pct(u['pose_coverage_2sigma'])} | "
                     f"{u['pose_nees_mean']:.2f} | {_pct(u['pose_marks']['endpoint']['coverage_2sigma'])} | "
                     f"{_pct(u['quality_fraction']['warn_or_alarm'])} |")
    det = result['detection']
    iid = result['datasets']['test_iid']['uncertainty']
    ood_u = result['datasets']['test_ood']['uncertainty']
    passed_velocity = .58 <= iid['coverage_1sigma'] <= .78
    passed_pose = iid['pose_coverage_2sigma'] >= .9
    lines += ['', '位姿 NEES 是二自由度，理想均值为 2；位姿 2σ 覆盖按卡方 95% 门限 5.99 统计，每秒取一个样本。',
              f"验收（test_iid）：速度 1σ 覆盖 {_pct(iid['coverage_1sigma'])}（目标 68% ± 10%）{'通过' if passed_velocity else '未通过'}；"
              f"位姿 2σ 覆盖 {_pct(iid['pose_coverage_2sigma'])}（目标 ≥ 90%）{'通过' if passed_pose else '未通过'}。", '',
              f"异常检出：以每条轨迹的种子间分歧均值为分数，区分 test_ood 与 test_iid 的 AUROC = {det['auroc_ood_vs_iid']:.3f}；"
              f"在全部测试轨迹中检出高误差轨迹（速度误差高于 test_iid 的 P90，共 {det['high_error_episodes']} 条）的 AUROC = "
              f"{det['auroc_high_error_episodes']:.3f}。", '',
              f"分布外工况下协方差偏小（test_ood 位姿 2σ 覆盖 {_pct(ood_u['pose_coverage_2sigma'])}），"
              f"但 warn/alarm 帧占 {_pct(ood_u['quality_fraction']['warn_or_alarm'])}，test_iid 只有 {_pct(iid['quality_fraction']['warn_or_alarm'])}。"
              '协方差数值只在训练分布内可信；quality 为 warn/alarm 时，下游融合应降低里程计权重或只把它当作告警。', '',
              '## yaw 漂移敏感性（odometry，人为线性漂移，协方差使用同一漂移率）', '',
              '| 漂移 (°/min) | 终点均值 (cm) | 位姿 2σ 覆盖 |', '|---:|---:|---:|']
    for drift, row in result['yaw_drift_sensitivity'].items():
        lines.append(f"| {drift} | {_cm(row['endpoint_mean_m'])} | {_pct(row['uncertainty']['pose_coverage_2sigma'])} |")
    if result.get('online'):
        lines += ['', '## 90 秒在线回归（运行时接口，新种子）', '',
                  '| 场景 | 跌倒 | 终点 (cm) | ATE (cm) | 终点 1σ (cm) | 终点 NEES | warn+alarm |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for name, row in result['online']['scenarios'].items():
            lines.append(f"| {name} | {'是' if row['fell'] else '否'} | {_cm(row['endpoint_xy_m'])} | {_cm(row['ate_xy_m'])} | "
                         f"{_cm(row['endpoint_sigma_xy_m'])} | {row['endpoint_nees']:.2f} | "
                         f"{_pct(row['quality']['warn'] + row['quality']['alarm'])} |")
        lines += ['', '在线位姿协方差包含默认 1°/min 的 yaw 随机游走项，长距离直行时它主导终点 1σ；仿真 yaw 是理想值，因此在线 NEES 偏小（偏保守）。']
    lines += ['', '## 口径与边界', '',
              '- 窗口、均值来源与全部校准参数只用额外验证集（val 200 + stop_val 60）；测试集在锁定后评测一次。',
              '- 速度误差参考世界系 0.2 s 低通标签；位置误差参考实际骨盆位置，只做初始位姿对齐；跌倒段保留。',
              '- Huber(delta=0.1) 在残差约 0.005 m/s 时等价于均方误差。',
              '- 不确定度是仿真内的统计校准；真机步态、IMU 安装误差和时延不在训练分布内，届时应以种子间分歧告警为准并重新校准。',
              '- 复现：`.venv/bin/python -m g1_odom run --device cpu`。', '']
    (REPORT/'report.md').write_text('\n'.join(lines))
