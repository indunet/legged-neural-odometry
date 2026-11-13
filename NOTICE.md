# Notice

## License

The code, released models and documentation of this project are licensed under the
Apache License, Version 2.0 (see `LICENSE`).

## Third-party assets

The license above does **not** cover the Unitree G1 model and walk policy in `assets/unitree_g1/`.

Those files are byte-for-byte copies from
[unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) at commit
`1425b15f73bd4095f0df53709d7c389c3eb9e790` (`g1.xml`, the meshes it references,
`policy.onnx`, and `deploy.yaml`). The mjlab Python package is not included and is
not required to run the simulation. SHA256 values are in `configs/assets.lock.json`;
`scripts/fetch_assets.py` (`python -m g1_odom fetch`) checks them and does not download
anything. The pinned upstream checkout has no root LICENSE; this project does not
assign new license terms to its policy or robot assets.

Anyone redistributing these materials must keep their notices and terms.
