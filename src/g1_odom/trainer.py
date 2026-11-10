#!/usr/bin/env python3
"""Supervised training of the lightweight velocity estimator; exports ONNX.

    python -m learned.train --data learned_data/train --out learned_models/demo/tcn
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .training_data import load_dataset
from .training_data import (FEATURE_NAMES, FK_FEATURES, FRAME_RATE, JOINT_RATE, LABEL_NAMES,
                              LABEL_TAU, NO_ANKLE_JOINTS, WINDOW, label_columns)
from .model import build, parameter_count

FEATURE_SETS = {"all": tuple(range(len(FEATURE_NAMES))),
                "nofk": tuple(i for i in range(len(FEATURE_NAMES)) if i not in FK_FEATURES),
                # IMU + hip/knee joints only: no ankle angles (parallel linkage on the robot), no FK
                "noankle": tuple(i for i, n in enumerate(FEATURE_NAMES)
                                 if n.startswith(("gyro_", "accel_", "gravity_"))
                                 or n in {f"{k}_{j}" for k in ("q", "dq") for j in NO_ANKLE_JOINTS})}


class Windows:
    """All episodes concatenated; a sample is the window ending at a valid frame."""

    def __init__(self, episodes, feature_index, window, device="cpu", label_index=(0, 1, 2)):
        self.window = window
        self.features = np.concatenate([e["features"][:, feature_index] for e in episodes])
        self.labels = np.concatenate([e["label"][:, list(label_index)] for e in episodes]).astype(np.float32)
        ends, start = [], 0
        for e in episodes:
            n = len(e["features"])
            if n >= window:
                ends.append(np.arange(start + window - 1, start + n))
            start += n
        self.ends = np.concatenate(ends) if ends else np.zeros(0, dtype=int)
        self.device = device
        self._features = torch.from_numpy(self.features).to(device)
        self._labels = torch.from_numpy(self.labels).to(device)
        self._offsets = torch.arange(-window + 1, 1, device=device)

    def __len__(self):
        return len(self.ends)

    def batch(self, ends):
        ends = torch.from_numpy(np.asarray(ends)).to(self.device)
        return self._features[ends[:, None] + self._offsets], self._labels[ends]


def default_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def split_episodes(episodes, fraction, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, int(round(len(episodes) * fraction))) if len(episodes) > 1 else 0
    val = [episodes[i] for i in order[:n_val]]
    train = [episodes[i] for i in order[n_val:]]
    return train, val


def export_onnx(model, path, n_features, window, metadata):
    import onnx
    model = model.eval().cpu()
    dummy = torch.zeros(1, window, n_features)
    torch.onnx.export(model, dummy, str(path), input_names=["features"], output_names=["velocity"],
                      dynamic_axes={"features": {0: "batch"}, "velocity": {0: "batch"}},
                      opset_version=17, dynamo=False)
    proto = onnx.load(str(path))
    del proto.metadata_props[:]
    for key, value in metadata.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, str(value)
    onnx.save(proto, str(path))


def evaluate_loss(model, data, loss_fn, limit=60000, batch=4096, seed=0):
    rng = np.random.default_rng(seed)
    ends = data.ends if len(data) <= limit else rng.choice(data.ends, limit, replace=False)
    total, sq = 0., np.zeros(data.labels.shape[1])
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ends), batch):
            x, y = data.batch(ends[i:i + batch])
            pred = model(x)
            total += float(loss_fn(pred, y)) * len(y)
            sq += ((pred - y) ** 2).sum(0).cpu().numpy()
    model.train()
    n = max(1, len(ends))
    return total / n, np.sqrt(sq / n)


STATIC_SPEED = 0.03     # m/s, label speed below which a window counts as standing


def static_mask(data, outputs):
    """True for windows whose last-frame planar label speed is below STATIC_SPEED."""
    cols = [i for i, o in enumerate(outputs) if o in ("vx", "vy")]
    if not cols:
        return np.zeros(len(data.ends), dtype=bool)
    return np.linalg.norm(data.labels[data.ends][:, cols], axis=1) < STATIC_SPEED


def sample_ends(rng, ends, static, batch, boost=0.):
    """Window ends for one batch; standing windows are drawn (1 + boost) times as often.

    boost=0 is the plain uniform draw (identical random stream to the original code).
    """
    if boost <= 0. or static is None or not static.any() or static.all():
        return rng.choice(ends, batch)
    n_static, n_moving = int(static.sum()), int((~static).sum())
    k = int(rng.binomial(batch, (1. + boost) * n_static / ((1. + boost) * n_static + n_moving)))
    return np.concatenate([rng.choice(ends[static], k), rng.choice(ends[~static], batch - k)])


def stored_label_names(episode):
    raw = episode.get("label_names")
    if raw is None:
        return LABEL_NAMES
    return tuple(str(name) for name in np.asarray(raw).reshape(-1))


def adopt_state(model, state):
    """Copy a checkpoint into ``model``, zero-filling new output rows of the head."""
    own = model.state_dict()
    merged = {}
    for key, value in own.items():
        old = state.get(key)
        if old is None:
            merged[key] = value
            continue
        old = old.detach().to(dtype=value.dtype, device=value.device)
        if tuple(old.shape) == tuple(value.shape):
            merged[key] = old
            continue
        if key == "head.weight" and tuple(old.shape[1:]) == tuple(value.shape[1:]) and old.shape[0] < value.shape[0]:
            grown = torch.zeros_like(value)
            grown[:old.shape[0]] = old
            merged[key] = grown
            continue
        if key == "head.bias" and old.ndim == value.ndim and old.shape[0] < value.shape[0]:
            grown = torch.zeros_like(value)
            grown[:old.shape[0]] = old
            merged[key] = grown
            continue
        raise ValueError(f"cannot grow {key} from {tuple(old.shape)} to {tuple(value.shape)}")
    model.load_state_dict(merged)


def checkpoint_channels(checkpoint):
    """Convolution width stored on a checkpoint, or inferred from the first conv."""
    if checkpoint is None:
        return None
    if checkpoint.get("channels") is not None:
        return int(checkpoint["channels"])
    weight = checkpoint.get("state_dict", {}).get("conv.0.weight")
    if weight is None:
        return None
    return int(weight.shape[0])


def epoch_score(val_loss, val_rmse, outputs, select="loss", vz_cap=None):
    """Lower is better. ``planar`` is vx RMSE + vy RMSE, and rejects a vz over ``vz_cap``."""
    if select == "loss":
        return float(val_loss)
    if select != "planar":
        raise ValueError(f"unknown select {select}")
    names = list(outputs)
    rmse = [float(v) for v in val_rmse]
    score = rmse[names.index("vx")] + rmse[names.index("vy")]
    if vz_cap is not None and "vz" in names and rmse[names.index("vz")] > float(vz_cap):
        return float("inf")
    return score


def _choose_epoch(candidates, select, vz_cap):
    """Pick ``(val_loss, planar, vz, state)``.

    ``planar`` keeps the lowest vx + vy among epochs under ``vz_cap``. If none
    qualify, it keeps the epochs closest to the best vz, then the lowest planar.
    """
    if select == "loss":
        return min(candidates, key=lambda item: item[0])[3], True
    eligible = [item for item in candidates if vz_cap is None or item[2] <= float(vz_cap)]
    cap_met = vz_cap is None or bool(eligible)
    if not eligible:
        best_vz = min(item[2] for item in candidates)
        eligible = [item for item in candidates if item[2] <= best_vz * 1.02]
    return min(eligible, key=lambda item: item[1])[3], cap_met


def train(datasets, out_dir, arch="tcn", features="all", window=WINDOW, epochs=20,
          steps_per_epoch=400, batch=2048, lr=3e-3, seed=0, val_fraction=0.1,
          init=None, extra_episodes=None, progress=None, device=None, outputs=LABEL_NAMES,
          static_boost=0., grow_outputs=False, channels=None, select="loss", vz_cap=None):
    """Train one variant; returns the metrics dict. ``init``: checkpoint to fine-tune.

    ``static_boost``: standing windows (label speed < 0.03 m/s) are sampled (1 + boost)
    times as often; 0 keeps the original uniform sampling.

    ``extra_episodes``: (episodes, weight) pairs added to training, e.g. real-robot
    logs; weight repeats them in the window pool.

    ``grow_outputs`` copies ``init`` and zero-fills output rows that the checkpoint
    does not have. Without it, a checkpoint keeps its own output list.

    ``channels`` widens a TCN. A different width than ``init`` keeps that
    checkpoint's feature mean and std and trains the weights from scratch.
    ``select="planar"`` keeps the epoch with the lowest vx + vy validation RMSE.
    ``vz_cap`` drops epochs whose vz RMSE is above the cap from that choice.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = device or default_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(init, weights_only=False) if init is not None else None
    expand = False
    inherit_weights = False
    width = 24 if channels is None else int(channels)
    if checkpoint:
        arch, window, feature_index = checkpoint["arch"], checkpoint["window"], checkpoint["feature_index"]
        stored_width = checkpoint_channels(checkpoint) or 24
        if channels is None:
            width = stored_width
        inherit_weights = width == stored_width
        stored = tuple(checkpoint.get("outputs", LABEL_NAMES))
        requested = tuple(outputs)
        if inherit_weights and grow_outputs:
            if stored != requested[:len(stored)] or len(requested) <= len(stored):
                raise ValueError(f"cannot grow outputs {stored} into {requested}")
            outputs = requested
            expand = True
        elif inherit_weights:
            outputs = stored
    else:
        feature_index = list(FEATURE_SETS[features])
    episodes = [e for d in datasets for e in load_dataset(d)]
    train_eps, val_eps = split_episodes(episodes, val_fraction, seed) if episodes else ([], [])
    for extra, weight in extra_episodes or []:
        extra_train, extra_val = split_episodes(extra, val_fraction, seed)
        train_eps += extra_train * max(1, int(weight))
        val_eps += extra_val
    outputs = tuple(outputs)
    names = stored_label_names((train_eps + val_eps)[0]) if (train_eps or val_eps) else LABEL_NAMES
    for episode in train_eps + val_eps:
        if stored_label_names(episode) != names:
            raise ValueError(f"mixed label columns: {names} vs {stored_label_names(episode)}")
    label_index = label_columns(names, outputs)
    train_data = Windows(train_eps, feature_index, window, device, label_index)
    val_data = Windows(val_eps, feature_index, window, device, label_index) if val_eps else train_data

    if checkpoint:
        mean, std = np.array(checkpoint["mean"]), np.array(checkpoint["std"])
    else:
        mean = train_data.features.mean(0)
        std = np.maximum(train_data.features.std(0), 1e-2)
    model = build(arch, len(feature_index), mean, std, window, n_out=len(outputs), channels=width)
    if checkpoint and inherit_weights:
        if expand:
            adopt_state(model, checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    params = parameter_count(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=epochs * steps_per_epoch)
    loss_fn = nn.HuberLoss(delta=0.1)
    rng = np.random.default_rng(seed)
    history, candidates = [], []
    static = static_mask(train_data, outputs) if static_boost > 0 else None
    started = time.monotonic()
    vz_index = list(outputs).index("vz") if "vz" in outputs else None
    # Infrastructure addition only: preserve the upstream numerical recipe while
    # making interrupted epochs recoverable in this independent project.
    signature={"datasets":[str(Path(d).resolve()) for d in datasets],"seed":seed,
               "epochs":epochs,"steps":steps_per_epoch,"batch":batch,"lr":lr,
               "outputs":list(outputs),"static_boost":static_boost,"select":select,
               "train_windows":len(train_data),"val_windows":len(val_data)}
    state_file=out_dir/"training_state.pt"
    first_epoch=0
    if state_file.exists():
        saved=torch.load(state_file,map_location=device,weights_only=False)
        if saved['signature']!=signature:raise ValueError('resume configuration differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);rng.bit_generator.state=saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.startswith('cuda') and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device)
        history=saved['history'];candidates=saved['candidates'];first_epoch=saved['epoch']+1
    for epoch in range(first_epoch,epochs):
        running = 0.
        for _ in range(steps_per_epoch):
            x, y = train_data.batch(sample_ends(rng, train_data.ends, static, batch, static_boost))
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
        val_loss, val_rmse = evaluate_loss(model, val_data, loss_fn)
        row = {"epoch": epoch + 1, "train": running / steps_per_epoch, "val": val_loss,
               "val_rmse": val_rmse.tolist(), "seconds": time.monotonic() - started}
        history.append(row)
        planar = epoch_score(val_loss, val_rmse, outputs, "planar", None)
        vz = float(val_rmse[vz_index]) if vz_index is not None else 0.0
        candidates.append((float(val_loss), planar, vz,
                           {k: v.detach().clone() for k, v in model.state_dict().items()}))
        torch.save({'signature':signature,'epoch':epoch,'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,
                    'history':history,'candidates':candidates},out_dir/'training_state.tmp.pt')
        (out_dir/'training_state.tmp.pt').replace(state_file)
        (out_dir/'train_log.json').write_text(json.dumps(history,indent=1))
        if progress:
            progress(epoch + 1, epochs, history[-1])
    best_state, cap_met = _choose_epoch(candidates, select, vz_cap)
    model.load_state_dict(best_state)
    _, val_rmse = evaluate_loss(model, val_data, loss_fn, limit=200000)

    names = [FEATURE_NAMES[i] for i in feature_index]
    fresh = init is None or expand or not inherit_weights
    metrics = {"arch": arch, "features": features if fresh else "finetuned",
               "window": window, "channels": width, "parameters": params, "train_windows": len(train_data),
               "val_windows": len(val_data), "train_episodes": len(train_eps),
               "val_episodes": len(val_eps), "val_rmse": dict(zip(outputs, val_rmse.tolist())), "outputs": list(outputs),
               "datasets": [str(d) for d in datasets], "init": str(init) if init else None,
               "static_boost": static_boost, "lr": lr, "select": select, "vz_cap": vz_cap,
               "vz_cap_met": cap_met, "inherited_weights": bool(checkpoint and inherit_weights),
               "device": device, "seconds": time.monotonic() - started}
    metrics['selected_epoch']=min(history,key=lambda row:row['val'])['epoch']
    metrics['selection_validation_loss']=min(row['val'] for row in history)
    metrics['recipe']='frame Huber only; no drift refinement'
    model = model.cpu()
    torch.save({"arch": arch, "window": window, "channels": width, "feature_index": feature_index,
                "outputs": list(outputs), "mean": mean.tolist(),
                "std": std.tolist(), "state_dict": model.state_dict()}, out_dir / "checkpoint.pt")
    export_onnx(model, out_dir / "estimator.onnx", len(feature_index), window, {
        "feature_names": ",".join(names), "window": window, "frame_rate": FRAME_RATE,
        "joint_rate": JOINT_RATE, "label_names": ",".join(outputs), "label_tau": LABEL_TAU,
        "output_frame": "heading (gravity aligned, pelvis yaw)", "arch": arch, "parameters": params})
    (out_dir / "train_log.json").write_text(json.dumps(history, indent=1))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arch", choices=["tcn", "mlp"], default="tcn")
    p.add_argument("--features", choices=list(FEATURE_SETS), default="all")
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outputs", default=",".join(LABEL_NAMES), help="label subset, e.g. vx,vy")
    p.add_argument("--static-boost", type=float, default=0.,
                   help="sample standing windows (1 + boost) times as often")
    args = p.parse_args()

    def progress(epoch, total, h):
        print(f"epoch {epoch}/{total} train {h['train']:.5f} val {h['val']:.5f} "
              f"rmse {np.round(h['val_rmse'], 4)}", flush=True)
    metrics = train(args.data, args.out, args.arch, args.features, args.window, args.epochs,
                    args.steps, seed=args.seed, progress=progress, outputs=args.outputs.split(","),
                    static_boost=args.static_boost)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
