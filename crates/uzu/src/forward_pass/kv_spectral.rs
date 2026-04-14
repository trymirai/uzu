use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::language_model::KvDebugSnapshot;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvSpectralLayerCalibration {
    pub layer_index: usize,
    pub num_groups: usize,
    pub head_dim: usize,
    pub key_basis: Box<[f32]>,
    pub key_variances: Box<[f32]>,
    pub value_basis: Box<[f32]>,
    pub value_variances: Box<[f32]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvSpectralCalibration {
    pub layers: Box<[KvSpectralLayerCalibration]>,
}

impl KvSpectralCalibration {
    pub fn from_snapshot(snapshot: &KvDebugSnapshot) -> Self {
        let layers = snapshot
            .layers
            .iter()
            .map(|layer| {
                let token_count = layer.positions.len();
                let (key_basis, key_variances) =
                    calibrate_tensor(&layer.keys, layer.num_groups, token_count, layer.head_dim);
                let (value_basis, value_variances) =
                    calibrate_tensor(&layer.values, layer.num_groups, token_count, layer.head_dim);
                KvSpectralLayerCalibration {
                    layer_index: layer.layer_index,
                    num_groups: layer.num_groups,
                    head_dim: layer.head_dim,
                    key_basis,
                    key_variances,
                    value_basis,
                    value_variances,
                }
            })
            .collect();
        Self {
            layers,
        }
    }

    pub fn load(path: &Path) -> Self {
        serde_json::from_str(&fs::read_to_string(path).expect("failed to read spectral calibration file"))
            .expect("failed to parse spectral calibration file")
    }

    pub fn save(
        &self,
        path: &Path,
    ) {
        fs::write(path, serde_json::to_string_pretty(self).expect("failed to serialize spectral calibration"))
            .expect("failed to write spectral calibration file");
    }

    pub fn layer(
        &self,
        layer_index: usize,
    ) -> &KvSpectralLayerCalibration {
        self.layers
            .iter()
            .find(|layer| layer.layer_index == layer_index)
            .expect("missing spectral calibration for transformer layer")
    }
}

pub fn dims_for_cumulative_variance(
    variances: &[f32],
    target: f32,
) -> usize {
    assert!((0.0..=1.0).contains(&target), "cumulative variance target must be in [0, 1]");
    let total = variances.iter().sum::<f32>();
    assert!(total > 0.0, "variance sum must be positive");
    let threshold = total * target;
    let mut cumulative = 0.0f32;
    for (index, &value) in variances.iter().enumerate() {
        cumulative += value;
        if cumulative >= threshold {
            return index + 1;
        }
    }
    variances.len()
}

pub fn average_group_variances(
    variances: &[f32],
    num_groups: usize,
    head_dim: usize,
) -> Box<[f32]> {
    let mut average = vec![0.0f32; head_dim];
    for group_index in 0..num_groups {
        let group_start = group_index * head_dim;
        for dim in 0..head_dim {
            average[dim] += variances[group_start + dim];
        }
    }
    for value in &mut average {
        *value /= num_groups as f32;
    }
    average.into_boxed_slice()
}

fn calibrate_tensor(
    flat: &[f32],
    num_groups: usize,
    token_count: usize,
    head_dim: usize,
) -> (Box<[f32]>, Box<[f32]>) {
    assert_eq!(flat.len(), num_groups * token_count * head_dim, "spectral calibration tensor shape mismatch");

    let mut basis = vec![0.0f32; num_groups * head_dim * head_dim];
    let mut variances = vec![0.0f32; num_groups * head_dim];

    for group_index in 0..num_groups {
        let group_start = group_index * token_count * head_dim;
        let group_end = group_start + token_count * head_dim;
        let (group_basis, group_variances) = calibrate_group(&flat[group_start..group_end], token_count, head_dim);

        let basis_start = group_index * head_dim * head_dim;
        let basis_end = basis_start + head_dim * head_dim;
        basis[basis_start..basis_end].copy_from_slice(&group_basis);

        let variance_start = group_index * head_dim;
        let variance_end = variance_start + head_dim;
        variances[variance_start..variance_end].copy_from_slice(&group_variances);
    }

    (basis.into_boxed_slice(), variances.into_boxed_slice())
}

fn calibrate_group(
    flat: &[f32],
    token_count: usize,
    head_dim: usize,
) -> (Box<[f32]>, Box<[f32]>) {
    let mut covariance = vec![0.0f64; head_dim * head_dim];
    let mut normalized = vec![0.0f64; head_dim];
    let mut row_count = 0usize;

    for row in flat.chunks_exact(head_dim).take(token_count) {
        let norm = row.iter().map(|&value| value as f64 * value as f64).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }

        for dim in 0..head_dim {
            normalized[dim] = row[dim] as f64 / norm;
        }

        for left in 0..head_dim {
            for right in left..head_dim {
                covariance[left * head_dim + right] += normalized[left] * normalized[right];
            }
        }

        row_count += 1;
    }

    assert!(row_count > 0, "spectral calibration needs at least one non-zero row");

    let scale = 1.0 / row_count as f64;
    for left in 0..head_dim {
        for right in left..head_dim {
            let value = covariance[left * head_dim + right] * scale;
            covariance[left * head_dim + right] = value;
            covariance[right * head_dim + left] = value;
        }
    }

    symmetric_eigendecomposition(covariance, head_dim)
}

fn symmetric_eigendecomposition(
    mut matrix: Vec<f64>,
    dim: usize,
) -> (Box<[f32]>, Box<[f32]>) {
    let mut eigenvectors = vec![0.0f64; dim * dim];
    for index in 0..dim {
        eigenvectors[index * dim + index] = 1.0;
    }

    for _ in 0..(dim * dim * 16) {
        let (p, q, off_diagonal) = largest_off_diagonal(&matrix, dim);
        if off_diagonal < 1e-12 {
            break;
        }

        let app = matrix[p * dim + p];
        let aqq = matrix[q * dim + q];
        let apq = matrix[p * dim + q];
        let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
        let cosine = theta.cos();
        let sine = theta.sin();

        for index in 0..dim {
            if index == p || index == q {
                continue;
            }
            let aip = matrix[index * dim + p];
            let aiq = matrix[index * dim + q];
            let next_ip = cosine * aip - sine * aiq;
            let next_iq = sine * aip + cosine * aiq;
            matrix[index * dim + p] = next_ip;
            matrix[p * dim + index] = next_ip;
            matrix[index * dim + q] = next_iq;
            matrix[q * dim + index] = next_iq;
        }

        matrix[p * dim + p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q * dim + q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p * dim + q] = 0.0;
        matrix[q * dim + p] = 0.0;

        for index in 0..dim {
            let vip = eigenvectors[index * dim + p];
            let viq = eigenvectors[index * dim + q];
            eigenvectors[index * dim + p] = cosine * vip - sine * viq;
            eigenvectors[index * dim + q] = sine * vip + cosine * viq;
        }
    }

    let mut order = (0..dim).collect::<Vec<_>>();
    order.sort_by(|&left, &right| matrix[right * dim + right].total_cmp(&matrix[left * dim + left]));

    let variances =
        order.iter().map(|&index| matrix[index * dim + index].max(0.0) as f32).collect::<Vec<_>>().into_boxed_slice();
    let mut basis = vec![0.0f32; dim * dim];
    for (component_index, &component) in order.iter().enumerate() {
        for input_dim in 0..dim {
            basis[component_index * dim + input_dim] = eigenvectors[input_dim * dim + component] as f32;
        }
    }

    (basis.into_boxed_slice(), variances)
}

fn largest_off_diagonal(
    matrix: &[f64],
    dim: usize,
) -> (usize, usize, f64) {
    let mut best_p = 0usize;
    let mut best_q = 1usize;
    let mut best_value = 0.0f64;

    for row in 0..dim {
        for col in row + 1..dim {
            let value = matrix[row * dim + col].abs();
            if value > best_value {
                best_value = value;
                best_p = row;
                best_q = col;
            }
        }
    }

    (best_p, best_q, best_value)
}
