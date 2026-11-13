//! Rotation helpers that follow SciPy 1.15 `Rotation.from_rotvec` and the
//! single-vector path of `Rotation.align_vectors`.

#[derive(Clone, Copy, Debug)]
pub struct Mat3(pub [[f64; 3]; 3]);

impl Mat3 {
    pub fn identity() -> Self {
        Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    pub fn mul(self, other: Self) -> Self {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = self.0[i][0] * other.0[0][j]
                    + self.0[i][1] * other.0[1][j]
                    + self.0[i][2] * other.0[2][j];
            }
        }
        Self(out)
    }

    /// `R.T @ v`.
    pub fn transpose_mul_vec(self, v: [f64; 3]) -> [f64; 3] {
        [
            self.0[0][0] * v[0] + self.0[1][0] * v[1] + self.0[2][0] * v[2],
            self.0[0][1] * v[0] + self.0[1][1] * v[1] + self.0[2][1] * v[2],
            self.0[0][2] * v[0] + self.0[1][2] * v[1] + self.0[2][2] * v[2],
        ]
    }
}

fn norm(v: [f64; 3]) -> f64 {
    v[0].hypot(v[1]).hypot(v[2])
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn quat_to_mat(x: f64, y: f64, z: f64, w: f64) -> Mat3 {
    let (x2, y2, z2, w2) = (x * x, y * y, z * z, w * w);
    let (xy, zw, xz, yw, yz, xw) = (x * y, z * w, x * z, y * w, y * z, x * w);
    Mat3([
        [x2 - y2 - z2 + w2, 2.0 * (xy - zw), 2.0 * (xz + yw)],
        [2.0 * (xy + zw), -x2 + y2 - z2 + w2, 2.0 * (yz - xw)],
        [2.0 * (xz - yw), 2.0 * (yz + xw), -x2 - y2 + z2 + w2],
    ])
}

/// SciPy `Rotation.from_rotvec`, including the small-angle series.
pub fn from_rotvec(rv: [f64; 3]) -> Mat3 {
    let angle = norm(rv);
    let (scale_q, w) = if angle <= 1e-3 {
        let angle2 = angle * angle;
        (
            0.5 - angle2 / 48.0 + angle2 * angle2 / 3840.0,
            (angle / 2.0).cos(),
        )
    } else {
        ((angle / 2.0).sin() / angle, (angle / 2.0).cos())
    };
    quat_to_mat(scale_q * rv[0], scale_q * rv[1], scale_q * rv[2], w)
}

/// Shortest rotation `R` such that `R @ up == [0, 0, 1]`, matching
/// `Rotation.align_vectors([[0, 0, 1]], [up])`.
pub fn align_up_to_world(up: [f64; 3]) -> Mat3 {
    let length = norm(up);
    let b = scale(up, 1.0 / length);
    let a = [0.0, 0.0, 1.0];
    let mut axis = cross(b, a);
    let mut theta = norm(axis).atan2(dot(a, b));
    let tolerance = 1e-3;
    let mut flip = Mat3::identity();
    if (std::f64::consts::PI - theta) < tolerance {
        let seed = if norm(axis) == 0.0 {
            antiparallel_axis(a)
        } else {
            axis
        };
        let seed_len = norm(seed);
        flip = from_rotvec(scale(seed, std::f64::consts::PI / seed_len));
        theta = std::f64::consts::PI - theta;
        axis = scale(axis, -1.0);
    }
    let rotvec = if theta.abs() < tolerance {
        let theta2 = theta * theta;
        scale(axis, 1.0 + theta2 / 6.0 + theta2 * theta2 * 7.0 / 360.0)
    } else {
        scale(axis, theta / theta.sin())
    };
    from_rotvec(rotvec).mul(flip)
}

/// Same branch SciPy uses when the two vectors are exactly opposite.
fn antiparallel_axis(a: [f64; 3]) -> [f64; 3] {
    let i = (0..3)
        .min_by(|&i, &j| a[i].abs().partial_cmp(&a[j].abs()).unwrap())
        .unwrap();
    let mut r = [0.0; 3];
    let i1 = (i + 2) % 3;
    let i2 = (i + 1) % 3;
    // Python `r[i - 1], r[i - 2] = a[i - 2], -a[i - 1]` with wraparound.
    r[i1] = a[i2];
    r[i2] = -a[i1];
    r
}
