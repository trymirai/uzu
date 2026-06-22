use serde::{Deserialize, Serialize};

const BF16_BYTES: f64 = 2.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coefficients {
    pub chip: String,
    pub a_j_per_byte: f64,
    pub b_j_per_flop: f64,
    pub p_idle_w: f64,
    pub peak_bandwidth_bytes_per_s: f64,
    pub peak_flop_rate_per_s: f64,
    pub samples: usize,
    pub rms_error_j: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Work {
    pub bytes: f64,
    pub flops: f64,
}

impl Coefficients {
    pub fn energy_joules(
        &self,
        work: Work,
    ) -> f64 {
        self.a_j_per_byte * work.bytes + self.b_j_per_flop * work.flops
    }
}

pub fn gemm_work(
    m: u64,
    k: u64,
    n: u64,
) -> Work {
    let (m, k, n) = (m as f64, k as f64, n as f64);
    Work {
        bytes: BF16_BYTES * (m * k + k * n + m * n),
        flops: 2.0 * m * k * n,
    }
}

pub fn parse_mkn(label: &str) -> Option<(u64, u64, u64)> {
    Some((dimension(label, "M[")?, dimension(label, "K[")?, dimension(label, "N[")?))
}

fn dimension(
    label: &str,
    key: &str,
) -> Option<u64> {
    let start = label.find(key)? + key.len();
    let rest = &label[start..];
    let end = rest.find(']')?;
    rest[..end].parse().ok()
}

pub struct Row {
    pub bytes: f64,
    pub flops: f64,
    pub gpu_seconds: f64,
    pub duration_seconds: f64,
    pub energy_joules: f64,
}

pub fn fit(
    chip: String,
    rows: &[Row],
) -> Option<Coefficients> {
    if rows.len() < 3 {
        return None;
    }
    let mut normal = [[0.0_f64; 3]; 3];
    let mut rhs = [0.0_f64; 3];
    for row in rows {
        let features = [row.bytes, row.flops, row.duration_seconds];
        for (i, fi) in features.iter().enumerate() {
            for (j, fj) in features.iter().enumerate() {
                normal[i][j] += fi * fj;
            }
            rhs[i] += fi * row.energy_joules;
        }
    }
    let [a, b, p_idle] = solve3(normal, rhs)?;

    let mut squared = 0.0;
    let mut peak_bandwidth = 0.0_f64;
    let mut peak_flop_rate = 0.0_f64;
    for row in rows {
        let predicted = a * row.bytes + b * row.flops + p_idle * row.duration_seconds;
        squared += (predicted - row.energy_joules).powi(2);
        if row.gpu_seconds > 0.0 {
            peak_bandwidth = peak_bandwidth.max(row.bytes / row.gpu_seconds);
            peak_flop_rate = peak_flop_rate.max(row.flops / row.gpu_seconds);
        }
    }
    Some(Coefficients {
        chip,
        a_j_per_byte: a,
        b_j_per_flop: b,
        p_idle_w: p_idle,
        peak_bandwidth_bytes_per_s: peak_bandwidth,
        peak_flop_rate_per_s: peak_flop_rate,
        samples: rows.len(),
        rms_error_j: (squared / rows.len() as f64).sqrt(),
    })
}

fn solve3(
    mut matrix: [[f64; 3]; 3],
    mut rhs: [f64; 3],
) -> Option<[f64; 3]> {
    for pivot in 0..3 {
        let mut best = pivot;
        for row in (pivot + 1)..3 {
            if matrix[row][pivot].abs() > matrix[best][pivot].abs() {
                best = row;
            }
        }
        if matrix[best][pivot].abs() < 1e-18 {
            return None;
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        for row in (pivot + 1)..3 {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            for column in pivot..3 {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut solution = [0.0_f64; 3];
    for row in (0..3).rev() {
        let mut value = rhs[row];
        for column in (row + 1)..3 {
            value -= matrix[row][column] * solution[column];
        }
        solution[row] = value / matrix[row][row];
    }
    Some(solution)
}
