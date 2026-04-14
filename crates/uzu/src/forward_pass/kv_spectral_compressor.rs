use std::f32::consts::PI;

use crate::{
    ArrayElement, DataType,
    array::Array,
    backends::common::Backend,
    forward_pass::{kv_cache_layer::KvCompressor, kv_spectral::KvSpectralLayerCalibration},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectralQuantTarget {
    Both,
    Keys,
    Values,
    Exact,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpectralValueLayoutSpec {
    Uniform {
        bits: usize,
    },
    MultiRegime {
        regimes: Box<[SpectralValueRegimeSpec]>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpectralValueRegimeSpec {
    pub end_dim: usize,
    pub bits: usize,
}

#[derive(Clone)]
enum SpectralValueLayout {
    Uniform {
        bits: usize,
        row_bytes: usize,
        codebook: Box<[f32]>,
    },
    MultiRegime {
        row_bytes: usize,
        regimes: Box<[SpectralValueRegime]>,
    },
}

#[derive(Clone)]
struct SpectralValueRegime {
    end_dim: usize,
    bits: usize,
    codebook: Box<[f32]>,
}

impl SpectralQuantTarget {
    pub(crate) fn quantize_keys(self) -> bool {
        matches!(self, SpectralQuantTarget::Both | SpectralQuantTarget::Keys)
    }

    pub(crate) fn quantize_values(self) -> bool {
        matches!(self, SpectralQuantTarget::Both | SpectralQuantTarget::Values)
    }
}

impl SpectralValueLayout {
    fn new(
        head_dim: usize,
        spec: SpectralValueLayoutSpec,
    ) -> Self {
        match spec {
            SpectralValueLayoutSpec::Uniform {
                bits,
            } => {
                assert!((1..=8).contains(&bits), "spectral value bits must be in [1, 8]");
                Self::Uniform {
                    bits,
                    row_bytes: (head_dim * bits).div_ceil(8),
                    codebook: gaussian_codebook(bits, head_dim).into_boxed_slice(),
                }
            },
            SpectralValueLayoutSpec::MultiRegime {
                regimes,
            } => {
                assert!(!regimes.is_empty(), "spectral value layout must have at least one regime");

                let mut previous_end = 0usize;
                let mut total_bits = 0usize;
                let regimes = regimes
                    .into_vec()
                    .into_iter()
                    .map(|regime| {
                        assert!((1..=8).contains(&regime.bits), "spectral value layout bits must be in [1, 8]");
                        assert!(regime.end_dim > previous_end, "spectral value layout end dims must be increasing");
                        assert!(regime.end_dim <= head_dim, "spectral value layout end dims must not exceed head dim");
                        total_bits += (regime.end_dim - previous_end) * regime.bits;
                        previous_end = regime.end_dim;
                        SpectralValueRegime {
                            end_dim: regime.end_dim,
                            bits: regime.bits,
                            codebook: gaussian_codebook(regime.bits, head_dim).into_boxed_slice(),
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice();

                assert_eq!(previous_end, head_dim, "spectral value layout must cover every rotated dim");

                Self::MultiRegime {
                    row_bytes: total_bits.div_ceil(8),
                    regimes,
                }
            },
        }
    }

    fn row_bytes(&self) -> usize {
        match self {
            SpectralValueLayout::Uniform {
                row_bytes,
                ..
            }
            | SpectralValueLayout::MultiRegime {
                row_bytes,
                ..
            } => *row_bytes,
        }
    }

    fn encode_row(
        &self,
        input: &[f32],
        packed: &mut [u8],
    ) {
        packed.fill(0);
        match self {
            SpectralValueLayout::Uniform {
                bits,
                codebook,
                ..
            } => encode_uniform_row(input, codebook, *bits, packed),
            SpectralValueLayout::MultiRegime {
                regimes,
                ..
            } => encode_multi_regime_row(input, regimes, packed),
        }
    }

    fn decode_row(
        &self,
        packed: &[u8],
        output: &mut [f32],
    ) {
        match self {
            SpectralValueLayout::Uniform {
                bits,
                codebook,
                ..
            } => decode_uniform_row(packed, codebook, *bits, output),
            SpectralValueLayout::MultiRegime {
                regimes,
                ..
            } => decode_multi_regime_row(packed, regimes, output),
        }
    }
}

pub struct SpectralQuantCompressor {
    shape: [usize; 3],
    target: SpectralQuantTarget,
    key_basis: Box<[f32]>,
    value_basis: Box<[f32]>,
    key_row_bytes: usize,
    value_layout: SpectralValueLayout,
    key_norms: Vec<f32>,
    key_signs: Vec<u8>,
    value_norms: Vec<f32>,
    value_codes: Vec<u8>,
    dense_row_bytes: usize,
    dense_data_type: Option<DataType>,
}

impl SpectralQuantCompressor {
    pub fn new(
        shape: [usize; 3],
        calibration: &KvSpectralLayerCalibration,
        value_layout: SpectralValueLayoutSpec,
        target: SpectralQuantTarget,
    ) -> Self {
        assert_eq!(calibration.num_groups, shape[0], "spectral calibration group count mismatch");
        assert_eq!(calibration.head_dim, shape[2], "spectral calibration head dim mismatch");

        let row_count = shape[0] * shape[1];
        let key_row_bytes = if target.quantize_keys() {
            shape[2].div_ceil(8)
        } else {
            0
        };
        let value_layout = SpectralValueLayout::new(shape[2], value_layout);
        let value_row_bytes = if target.quantize_values() {
            value_layout.row_bytes()
        } else {
            0
        };

        Self {
            shape,
            target,
            key_basis: calibration.key_basis.clone(),
            value_basis: calibration.value_basis.clone(),
            key_row_bytes,
            value_layout,
            key_norms: vec![0.0; row_count],
            key_signs: vec![0; row_count * key_row_bytes],
            value_norms: vec![0.0; row_count],
            value_codes: vec![0; row_count * value_row_bytes],
            dense_row_bytes: 0,
            dense_data_type: None,
        }
    }

    fn basis_for_group<'a>(
        basis: &'a [f32],
        group_index: usize,
        head_dim: usize,
    ) -> &'a [f32] {
        let group_stride = head_dim * head_dim;
        let start = group_index * group_stride;
        &basis[start..start + group_stride]
    }

    fn decode_value_row_into(
        &self,
        row_index: usize,
        scratch: &mut [f32],
        output: &mut [f32],
    ) {
        let head_dim = self.shape[2];
        let group_index = row_index / self.shape[1];
        let basis = Self::basis_for_group(&self.value_basis, group_index, head_dim);
        assert_eq!(scratch.len(), head_dim, "spectral value row scratch shape mismatch");
        assert_eq!(output.len(), head_dim, "spectral value row shape mismatch");
        self.value_layout.decode_row(
            &self.value_codes
                [row_index * self.value_layout.row_bytes()..(row_index + 1) * self.value_layout.row_bytes()],
            scratch,
        );
        inverse_rotate(basis, scratch, output);
        for value in output.iter_mut() {
            *value *= self.value_norms[row_index];
        }
    }

    fn ensure_dense_storage(
        &mut self,
        data_type: DataType,
    ) {
        if let Some(existing) = self.dense_data_type {
            assert_eq!(existing, data_type, "spectral dense storage dtype mismatch");
            return;
        }

        let row_count = self.shape[0] * self.shape[1];
        self.dense_row_bytes = self.shape[2] * data_type.size_in_bytes();
        self.dense_data_type = Some(data_type);
        let _ = row_count;
    }

    fn compress_keys_typed<T>(
        &mut self,
        keys: &Array<impl Backend>,
    ) where
        T: ArrayElement + Copy,
        f32: From<T>,
    {
        let head_dim = self.shape[2];
        let sequence_length = self.shape[1];
        let mut normalized = vec![0.0f32; head_dim];
        let mut rotated = vec![0.0f32; head_dim];

        for group_index in 0..self.shape[0] {
            let basis = Self::basis_for_group(&self.key_basis, group_index, head_dim);
            for token_index in 0..sequence_length {
                let row_index = group_index * sequence_length + token_index;
                let row = &keys.as_slice::<T>()[row_index * head_dim..(row_index + 1) * head_dim];
                let packed = &mut self.key_signs[row_index * self.key_row_bytes..(row_index + 1) * self.key_row_bytes];
                compress_sign_row(row, basis, packed, &mut self.key_norms[row_index], &mut normalized, &mut rotated);
            }
        }
    }

    fn compress_values_typed<T>(
        &mut self,
        values: &Array<impl Backend>,
    ) where
        T: ArrayElement + Copy,
        f32: From<T>,
    {
        let head_dim = self.shape[2];
        let sequence_length = self.shape[1];
        let mut normalized = vec![0.0f32; head_dim];
        let mut rotated = vec![0.0f32; head_dim];

        for group_index in 0..self.shape[0] {
            let basis = Self::basis_for_group(&self.value_basis, group_index, head_dim);
            for token_index in 0..sequence_length {
                let row_index = group_index * sequence_length + token_index;
                let row = &values.as_slice::<T>()[row_index * head_dim..(row_index + 1) * head_dim];
                let packed = &mut self.value_codes
                    [row_index * self.value_layout.row_bytes()..(row_index + 1) * self.value_layout.row_bytes()];
                compress_quantized_row(
                    row,
                    basis,
                    &self.value_layout,
                    packed,
                    &mut self.value_norms[row_index],
                    &mut normalized,
                    &mut rotated,
                );
            }
        }
    }

    fn update_rows_from_dense_typed<T>(
        &mut self,
        keys: &Array<impl Backend>,
        values: &Array<impl Backend>,
        source_indices: &[usize],
        destination_indices: &[usize],
    ) where
        T: ArrayElement + Copy,
        f32: From<T>,
    {
        assert_eq!(source_indices.len(), destination_indices.len(), "spectral row update index mismatch");

        if !self.target.quantize_keys() && !self.target.quantize_values() {
            return;
        }

        let head_dim = self.shape[2];
        let sequence_length = self.shape[1];
        let mut normalized = vec![0.0f32; head_dim];
        let mut rotated = vec![0.0f32; head_dim];

        for (&source_index, &destination_index) in source_indices.iter().zip(destination_indices.iter()) {
            assert!(source_index < sequence_length, "spectral source row out of bounds");
            assert!(destination_index < sequence_length, "spectral destination row out of bounds");
            for group_index in 0..self.shape[0] {
                let source_row_index = group_index * sequence_length + source_index;
                let destination_row_index = group_index * sequence_length + destination_index;

                if self.target.quantize_keys() {
                    let basis = Self::basis_for_group(&self.key_basis, group_index, head_dim);
                    let row = &keys.as_slice::<T>()[source_row_index * head_dim..(source_row_index + 1) * head_dim];
                    let packed = &mut self.key_signs
                        [destination_row_index * self.key_row_bytes..(destination_row_index + 1) * self.key_row_bytes];
                    compress_sign_row(
                        row,
                        basis,
                        packed,
                        &mut self.key_norms[destination_row_index],
                        &mut normalized,
                        &mut rotated,
                    );
                }

                if self.target.quantize_values() {
                    let basis = Self::basis_for_group(&self.value_basis, group_index, head_dim);
                    let row = &values.as_slice::<T>()[source_row_index * head_dim..(source_row_index + 1) * head_dim];
                    let packed = &mut self.value_codes[destination_row_index * self.value_layout.row_bytes()
                        ..(destination_row_index + 1) * self.value_layout.row_bytes()];
                    compress_quantized_row(
                        row,
                        basis,
                        &self.value_layout,
                        packed,
                        &mut self.value_norms[destination_row_index],
                        &mut normalized,
                        &mut rotated,
                    );
                }
            }
        }
    }

    fn decompress_keys_typed<T>(
        &self,
        array: &mut Array<impl Backend>,
    ) where
        T: ArrayElement + Copy,
        f32: From<T>,
    {
        let head_dim = self.shape[2];
        let sequence_length = self.shape[1];
        let mut rotated = vec![0.0f32; head_dim];
        let mut restored = vec![0.0f32; head_dim];

        for group_index in 0..self.shape[0] {
            let basis = Self::basis_for_group(&self.key_basis, group_index, head_dim);
            for token_index in 0..sequence_length {
                let row_index = group_index * sequence_length + token_index;
                decode_sign_row(
                    &self.key_signs[row_index * self.key_row_bytes..(row_index + 1) * self.key_row_bytes],
                    &mut rotated,
                );
                inverse_rotate(basis, &rotated, &mut restored);
                for (dst, &src) in array.as_slice_mut::<T>()[row_index * head_dim..(row_index + 1) * head_dim]
                    .iter_mut()
                    .zip(restored.iter())
                {
                    *dst = num_traits::cast(src * self.key_norms[row_index]).expect("spectral key cast must succeed");
                }
            }
        }
    }

    fn decompress_values_typed<T>(
        &self,
        array: &mut Array<impl Backend>,
    ) where
        T: ArrayElement + Copy,
        f32: From<T>,
    {
        let head_dim = self.shape[2];
        let sequence_length = self.shape[1];
        let mut rotated = vec![0.0f32; head_dim];
        let mut restored = vec![0.0f32; head_dim];

        for group_index in 0..self.shape[0] {
            let basis = Self::basis_for_group(&self.value_basis, group_index, head_dim);
            for token_index in 0..sequence_length {
                let row_index = group_index * sequence_length + token_index;
                self.value_layout.decode_row(
                    &self.value_codes
                        [row_index * self.value_layout.row_bytes()..(row_index + 1) * self.value_layout.row_bytes()],
                    &mut rotated,
                );
                inverse_rotate(basis, &rotated, &mut restored);
                for (dst, &src) in array.as_slice_mut::<T>()[row_index * head_dim..(row_index + 1) * head_dim]
                    .iter_mut()
                    .zip(restored.iter())
                {
                    *dst =
                        num_traits::cast(src * self.value_norms[row_index]).expect("spectral value cast must succeed");
                }
            }
        }
    }
}

impl<B: Backend> KvCompressor<B> for SpectralQuantCompressor {
    fn compress(
        &mut self,
        keys: &Array<B>,
        values: &Array<B>,
    ) {
        assert_eq!(keys.shape(), &self.shape, "spectral key shape mismatch");
        assert_eq!(values.shape(), &self.shape, "spectral value shape mismatch");
        assert_eq!(keys.data_type(), values.data_type(), "spectral KV dtype mismatch");
        self.ensure_dense_storage(keys.data_type());

        match keys.data_type() {
            DataType::BF16 => {
                if self.target.quantize_keys() {
                    self.compress_keys_typed::<half::bf16>(keys);
                }
                if self.target.quantize_values() {
                    self.compress_values_typed::<half::bf16>(values);
                }
            },
            DataType::F16 => {
                if self.target.quantize_keys() {
                    self.compress_keys_typed::<half::f16>(keys);
                }
                if self.target.quantize_values() {
                    self.compress_values_typed::<half::f16>(values);
                }
            },
            DataType::F32 => {
                if self.target.quantize_keys() {
                    self.compress_keys_typed::<f32>(keys);
                }
                if self.target.quantize_values() {
                    self.compress_values_typed::<f32>(values);
                }
            },
            dtype => panic!("spectral quant does not support KV dtype {dtype:?}"),
        }
    }

    fn decompress(
        &self,
        keys: &mut Array<B>,
        values: &mut Array<B>,
    ) {
        assert_eq!(keys.shape(), &self.shape, "spectral key shape mismatch");
        assert_eq!(values.shape(), &self.shape, "spectral value shape mismatch");
        assert_eq!(keys.data_type(), values.data_type(), "spectral KV dtype mismatch");

        match keys.data_type() {
            DataType::BF16 => {
                if self.target.quantize_keys() {
                    self.decompress_keys_typed::<half::bf16>(keys);
                }
                if self.target.quantize_values() {
                    self.decompress_values_typed::<half::bf16>(values);
                }
            },
            DataType::F16 => {
                if self.target.quantize_keys() {
                    self.decompress_keys_typed::<half::f16>(keys);
                }
                if self.target.quantize_values() {
                    self.decompress_values_typed::<half::f16>(values);
                }
            },
            DataType::F32 => {
                if self.target.quantize_keys() {
                    self.decompress_keys_typed::<f32>(keys);
                }
                if self.target.quantize_values() {
                    self.decompress_values_typed::<f32>(values);
                }
            },
            dtype => panic!("spectral quant does not support KV dtype {dtype:?}"),
        }
    }

    fn decompress_keys(
        &self,
        keys: &mut Array<B>,
    ) {
        assert_eq!(keys.shape(), &self.shape, "spectral key shape mismatch");
        match keys.data_type() {
            DataType::BF16 => {
                if self.target.quantize_keys() {
                    self.decompress_keys_typed::<half::bf16>(keys);
                }
            },
            DataType::F16 => {
                if self.target.quantize_keys() {
                    self.decompress_keys_typed::<half::f16>(keys);
                }
            },
            DataType::F32 => {
                if self.target.quantize_keys() {
                    self.decompress_keys_typed::<f32>(keys);
                }
            },
            dtype => panic!("spectral quant does not support KV dtype {dtype:?}"),
        }
    }

    fn decompress_values(
        &self,
        values: &mut Array<B>,
    ) {
        assert_eq!(values.shape(), &self.shape, "spectral value shape mismatch");
        match values.data_type() {
            DataType::BF16 => {
                if self.target.quantize_values() {
                    self.decompress_values_typed::<half::bf16>(values);
                }
            },
            DataType::F16 => {
                if self.target.quantize_values() {
                    self.decompress_values_typed::<half::f16>(values);
                }
            },
            DataType::F32 => {
                if self.target.quantize_values() {
                    self.decompress_values_typed::<f32>(values);
                }
            },
            dtype => panic!("spectral quant does not support KV dtype {dtype:?}"),
        }
    }

    fn update_rows_from_dense(
        &mut self,
        _context: &B::Context,
        keys: &Array<B>,
        values: &Array<B>,
        source_indices: &[usize],
        destination_indices: &[usize],
    ) {
        assert_eq!(keys.shape(), &self.shape, "spectral key shape mismatch");
        assert_eq!(values.shape(), &self.shape, "spectral value shape mismatch");
        assert_eq!(keys.data_type(), values.data_type(), "spectral KV dtype mismatch");
        self.ensure_dense_storage(keys.data_type());

        match keys.data_type() {
            DataType::BF16 => {
                self.update_rows_from_dense_typed::<half::bf16>(keys, values, source_indices, destination_indices)
            },
            DataType::F16 => {
                self.update_rows_from_dense_typed::<half::f16>(keys, values, source_indices, destination_indices)
            },
            DataType::F32 => {
                self.update_rows_from_dense_typed::<f32>(keys, values, source_indices, destination_indices)
            },
            dtype => panic!("spectral quant does not support KV dtype {dtype:?}"),
        }
    }

    fn fill_prefix_attention_scores_for_single_decode(
        &self,
        queries: &[f32],
        num_heads: usize,
        prefix_length: usize,
        scores_out: &mut [f32],
    ) -> bool {
        if !self.target.quantize_keys() {
            return false;
        }

        let num_groups = self.shape[0];
        let sequence_length = self.shape[1];
        let head_dim = self.shape[2];
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        assert_eq!(queries.len(), num_heads * head_dim, "spectral query shape mismatch");
        assert_eq!(scores_out.len(), num_heads * prefix_length, "spectral score shape mismatch");
        assert_eq!(num_heads % num_groups, 0, "spectral GQA factor must divide head count");
        assert!(prefix_length <= sequence_length, "spectral prefix length out of bounds");

        let gqa_factor = num_heads / num_groups;
        let mut rotated_query = vec![0.0f32; head_dim];

        for head_index in 0..num_heads {
            let query = &queries[head_index * head_dim..(head_index + 1) * head_dim];
            let group_index = head_index / gqa_factor;
            rotate(Self::basis_for_group(&self.key_basis, group_index, head_dim), query, &mut rotated_query);

            for token_index in 0..prefix_length {
                let row_index = group_index * sequence_length + token_index;
                scores_out[head_index * prefix_length + token_index] = self.key_norms[row_index]
                    * dot_sign_row(
                        &self.key_signs[row_index * self.key_row_bytes..(row_index + 1) * self.key_row_bytes],
                        &rotated_query,
                        scale,
                    );
            }
        }

        true
    }

    fn supports_prefix_attention_scores_for_single_decode(&self) -> bool {
        self.target.quantize_keys()
    }

    fn supports_value_row_decoding_for_single_decode(&self) -> bool {
        self.target.quantize_values()
    }

    fn decode_value_row_for_single_decode(
        &self,
        row_index: usize,
        scratch: &mut [f32],
        output: &mut [f32],
    ) {
        assert!(self.target.quantize_values(), "spectral quant value row decoding requires quantized values");
        assert!(row_index < self.shape[0] * self.shape[1], "spectral quant value row index out of bounds");
        assert_eq!(output.len(), self.shape[2], "spectral quant value row shape mismatch");
        self.decode_value_row_into(row_index, scratch, output);
    }

    fn decode_value_rows_for_single_decode(
        &self,
        row_index: usize,
        row_count: usize,
        output: &mut [f32],
    ) {
        if row_count == 0 {
            assert!(output.is_empty(), "spectral quant row-range output must be empty when row_count is zero");
            return;
        }
        assert!(self.target.quantize_values(), "spectral quant value row decoding requires quantized values");
        assert!(row_index + row_count <= self.shape[0] * self.shape[1], "spectral quant row range out of bounds");
        let head_dim = self.shape[2];
        assert_eq!(output.len(), row_count * head_dim, "spectral quant row-range output shape mismatch");

        let sequence_length = self.shape[1];
        let group_index = row_index / sequence_length;
        assert_eq!(
            (row_index + row_count - 1) / sequence_length,
            group_index,
            "spectral quant row-range decode must stay within one group"
        );
        let basis = Self::basis_for_group(&self.value_basis, group_index, head_dim);
        let row_bytes = self.value_layout.row_bytes();
        let mut scratch = vec![0.0f32; head_dim];

        for (offset, row) in output.chunks_exact_mut(head_dim).enumerate() {
            let row_index = row_index + offset;
            self.value_layout
                .decode_row(&self.value_codes[row_index * row_bytes..(row_index + 1) * row_bytes], &mut scratch);
            inverse_rotate(basis, &scratch, row);
            let norm = self.value_norms[row_index];
            for value in row.iter_mut() {
                *value *= norm;
            }
        }
    }

    fn memory_usage_bytes(&self) -> usize {
        let key_storage = if self.target.quantize_keys() {
            self.key_signs.len() + self.key_norms.len() * std::mem::size_of::<f32>()
        } else {
            0
        };
        let value_storage = if self.target.quantize_values() {
            self.value_codes.len() + self.value_norms.len() * std::mem::size_of::<f32>()
        } else {
            0
        };
        key_storage + value_storage
    }
}

pub struct ShapedCacheCompressor {
    shape: [usize; 3],
    target: SpectralQuantTarget,
    key_basis: Box<[f32]>,
    value_basis: Box<[f32]>,
    key_rank: usize,
    value_rank: usize,
    key_norms: Vec<f32>,
    key_coefficients: Vec<half::bf16>,
    value_norms: Vec<f32>,
    value_coefficients: Vec<half::bf16>,
}

impl ShapedCacheCompressor {
    pub fn new(
        shape: [usize; 3],
        calibration: &KvSpectralLayerCalibration,
        key_rank: usize,
        value_rank: usize,
        target: SpectralQuantTarget,
    ) -> Self {
        assert_eq!(calibration.num_groups, shape[0], "shaped cache calibration group count mismatch");
        assert_eq!(calibration.head_dim, shape[2], "shaped cache calibration head dim mismatch");
        assert!((1..=shape[2]).contains(&key_rank), "shaped cache key rank must be in [1, head_dim]");
        assert!((1..=shape[2]).contains(&value_rank), "shaped cache value rank must be in [1, head_dim]");

        let row_count = shape[0] * shape[1];
        let key_coefficients = if target.quantize_keys() {
            vec![half::bf16::ZERO; row_count * key_rank]
        } else {
            Vec::new()
        };
        let value_coefficients = if target.quantize_values() {
            vec![half::bf16::ZERO; row_count * value_rank]
        } else {
            Vec::new()
        };

        Self {
            shape,
            target,
            key_basis: calibration.key_basis.clone(),
            value_basis: calibration.value_basis.clone(),
            key_rank,
            value_rank,
            key_norms: vec![0.0; row_count],
            key_coefficients,
            value_norms: vec![0.0; row_count],
            value_coefficients,
        }
    }

    fn basis_for_group<'a>(
        basis: &'a [f32],
        group_index: usize,
        head_dim: usize,
    ) -> &'a [f32] {
        let group_stride = head_dim * head_dim;
        let start = group_index * group_stride;
        &basis[start..start + group_stride]
    }

    fn decode_value_row_into(
        &self,
        row_index: usize,
        scratch: &mut [f32],
        output: &mut [f32],
    ) {
        let head_dim = self.shape[2];
        let group_index = row_index / self.shape[1];
        let basis = Self::basis_for_group(&self.value_basis, group_index, head_dim);
        assert_eq!(scratch.len(), head_dim, "shaped cache value row scratch shape mismatch");
        assert_eq!(output.len(), head_dim, "shaped cache value row shape mismatch");
        decode_truncated_row(
            &self.value_coefficients[row_index * self.value_rank..(row_index + 1) * self.value_rank],
            scratch,
        );
        inverse_rotate(basis, scratch, output);
        for value in output.iter_mut() {
            *value *= self.value_norms[row_index];
        }
    }

    fn compress_keys_typed<T>(
        &mut self,
        keys: &Array<impl Backend>,
    ) where
        T: ArrayElement + Copy,
        f32: From<T>,
    {
        let head_dim = self.shape[2];
        let sequence_length = self.shape[1];
        let mut normalized = vec![0.0f32; head_dim];
        let mut rotated = vec![0.0f32; head_dim];

        for group_index in 0..self.shape[0] {
            let basis = Self::basis_for_group(&self.key_basis, group_index, head_dim);
            for token_index in 0..sequence_length {
                let row_index = group_index * sequence_length + token_index;
                let row = &keys.as_slice::<T>()[row_index * head_dim..(row_index + 1) * head_dim];
                let stored = &mut self.key_coefficients[row_index * self.key_rank..(row_index + 1) * self.key_rank];
                compress_truncated_row(
                    row,
                    basis,
                    self.key_rank,
                    stored,
                    &mut self.key_norms[row_index],
                    &mut normalized,
                    &mut rotated,
                );
            }
        }
    }

    fn compress_values_typed<T>(
        &mut self,
        values: &Array<impl Backend>,
    ) where
        T: ArrayElement + Copy,
        f32: From<T>,
    {
        let head_dim = self.shape[2];
        let sequence_length = self.shape[1];
        let mut normalized = vec![0.0f32; head_dim];
        let mut rotated = vec![0.0f32; head_dim];

        for group_index in 0..self.shape[0] {
            let basis = Self::basis_for_group(&self.value_basis, group_index, head_dim);
            for token_index in 0..sequence_length {
                let row_index = group_index * sequence_length + token_index;
                let row = &values.as_slice::<T>()[row_index * head_dim..(row_index + 1) * head_dim];
                let stored =
                    &mut self.value_coefficients[row_index * self.value_rank..(row_index + 1) * self.value_rank];
                compress_truncated_row(
                    row,
                    basis,
                    self.value_rank,
                    stored,
                    &mut self.value_norms[row_index],
                    &mut normalized,
                    &mut rotated,
                );
            }
        }
    }

    fn update_rows_from_dense_typed<T>(
        &mut self,
        keys: &Array<impl Backend>,
        values: &Array<impl Backend>,
        source_indices: &[usize],
        destination_indices: &[usize],
    ) where
        T: ArrayElement + Copy,
        f32: From<T>,
    {
        assert_eq!(source_indices.len(), destination_indices.len(), "shaped cache row update index mismatch");

        if !self.target.quantize_keys() && !self.target.quantize_values() {
            return;
        }

        let head_dim = self.shape[2];
        let sequence_length = self.shape[1];
        let mut normalized = vec![0.0f32; head_dim];
        let mut rotated = vec![0.0f32; head_dim];

        for (&source_index, &destination_index) in source_indices.iter().zip(destination_indices.iter()) {
            assert!(source_index < sequence_length, "shaped cache source row out of bounds");
            assert!(destination_index < sequence_length, "shaped cache destination row out of bounds");

            for group_index in 0..self.shape[0] {
                let source_row_index = group_index * sequence_length + source_index;
                let destination_row_index = group_index * sequence_length + destination_index;

                if self.target.quantize_keys() {
                    let basis = Self::basis_for_group(&self.key_basis, group_index, head_dim);
                    let row = &keys.as_slice::<T>()[source_row_index * head_dim..(source_row_index + 1) * head_dim];
                    let stored = &mut self.key_coefficients
                        [destination_row_index * self.key_rank..(destination_row_index + 1) * self.key_rank];
                    compress_truncated_row(
                        row,
                        basis,
                        self.key_rank,
                        stored,
                        &mut self.key_norms[destination_row_index],
                        &mut normalized,
                        &mut rotated,
                    );
                }

                if self.target.quantize_values() {
                    let basis = Self::basis_for_group(&self.value_basis, group_index, head_dim);
                    let row = &values.as_slice::<T>()[source_row_index * head_dim..(source_row_index + 1) * head_dim];
                    let stored = &mut self.value_coefficients
                        [destination_row_index * self.value_rank..(destination_row_index + 1) * self.value_rank];
                    compress_truncated_row(
                        row,
                        basis,
                        self.value_rank,
                        stored,
                        &mut self.value_norms[destination_row_index],
                        &mut normalized,
                        &mut rotated,
                    );
                }
            }
        }
    }

    fn decompress_keys_typed<T>(
        &self,
        array: &mut Array<impl Backend>,
    ) where
        T: ArrayElement + Copy,
        f32: From<T>,
    {
        let head_dim = self.shape[2];
        let sequence_length = self.shape[1];
        let mut rotated = vec![0.0f32; head_dim];
        let mut restored = vec![0.0f32; head_dim];

        for group_index in 0..self.shape[0] {
            let basis = Self::basis_for_group(&self.key_basis, group_index, head_dim);
            for token_index in 0..sequence_length {
                let row_index = group_index * sequence_length + token_index;
                decode_truncated_row(
                    &self.key_coefficients[row_index * self.key_rank..(row_index + 1) * self.key_rank],
                    &mut rotated,
                );
                inverse_rotate(basis, &rotated, &mut restored);
                for (dst, &src) in array.as_slice_mut::<T>()[row_index * head_dim..(row_index + 1) * head_dim]
                    .iter_mut()
                    .zip(restored.iter())
                {
                    *dst =
                        num_traits::cast(src * self.key_norms[row_index]).expect("shaped cache key cast must succeed");
                }
            }
        }
    }

    fn decompress_values_typed<T>(
        &self,
        array: &mut Array<impl Backend>,
    ) where
        T: ArrayElement + Copy,
        f32: From<T>,
    {
        let head_dim = self.shape[2];
        let sequence_length = self.shape[1];
        let mut rotated = vec![0.0f32; head_dim];
        let mut restored = vec![0.0f32; head_dim];

        for group_index in 0..self.shape[0] {
            let basis = Self::basis_for_group(&self.value_basis, group_index, head_dim);
            for token_index in 0..sequence_length {
                let row_index = group_index * sequence_length + token_index;
                decode_truncated_row(
                    &self.value_coefficients[row_index * self.value_rank..(row_index + 1) * self.value_rank],
                    &mut rotated,
                );
                inverse_rotate(basis, &rotated, &mut restored);
                for (dst, &src) in array.as_slice_mut::<T>()[row_index * head_dim..(row_index + 1) * head_dim]
                    .iter_mut()
                    .zip(restored.iter())
                {
                    *dst = num_traits::cast(src * self.value_norms[row_index])
                        .expect("shaped cache value cast must succeed");
                }
            }
        }
    }
}

impl<B: Backend> KvCompressor<B> for ShapedCacheCompressor {
    fn compress(
        &mut self,
        keys: &Array<B>,
        values: &Array<B>,
    ) {
        assert_eq!(keys.shape(), &self.shape, "shaped cache key shape mismatch");
        assert_eq!(values.shape(), &self.shape, "shaped cache value shape mismatch");
        assert_eq!(keys.data_type(), values.data_type(), "shaped cache KV dtype mismatch");

        match keys.data_type() {
            DataType::BF16 => {
                if self.target.quantize_keys() {
                    self.compress_keys_typed::<half::bf16>(keys);
                }
                if self.target.quantize_values() {
                    self.compress_values_typed::<half::bf16>(values);
                }
            },
            DataType::F16 => {
                if self.target.quantize_keys() {
                    self.compress_keys_typed::<half::f16>(keys);
                }
                if self.target.quantize_values() {
                    self.compress_values_typed::<half::f16>(values);
                }
            },
            DataType::F32 => {
                if self.target.quantize_keys() {
                    self.compress_keys_typed::<f32>(keys);
                }
                if self.target.quantize_values() {
                    self.compress_values_typed::<f32>(values);
                }
            },
            dtype => panic!("shaped cache does not support dtype {dtype:?}"),
        }
    }

    fn decompress(
        &self,
        keys: &mut Array<B>,
        values: &mut Array<B>,
    ) {
        assert_eq!(keys.shape(), &self.shape, "shaped cache key shape mismatch");
        assert_eq!(values.shape(), &self.shape, "shaped cache value shape mismatch");
        assert_eq!(keys.data_type(), values.data_type(), "shaped cache KV dtype mismatch");

        match keys.data_type() {
            DataType::BF16 => {
                if self.target.quantize_keys() {
                    self.decompress_keys_typed::<half::bf16>(keys);
                }
                if self.target.quantize_values() {
                    self.decompress_values_typed::<half::bf16>(values);
                }
            },
            DataType::F16 => {
                if self.target.quantize_keys() {
                    self.decompress_keys_typed::<half::f16>(keys);
                }
                if self.target.quantize_values() {
                    self.decompress_values_typed::<half::f16>(values);
                }
            },
            DataType::F32 => {
                if self.target.quantize_keys() {
                    self.decompress_keys_typed::<f32>(keys);
                }
                if self.target.quantize_values() {
                    self.decompress_values_typed::<f32>(values);
                }
            },
            dtype => panic!("shaped cache does not support dtype {dtype:?}"),
        }
    }

    fn decompress_keys(
        &self,
        keys: &mut Array<B>,
    ) {
        assert_eq!(keys.shape(), &self.shape, "shaped cache key shape mismatch");
        match keys.data_type() {
            DataType::BF16 => {
                if self.target.quantize_keys() {
                    self.decompress_keys_typed::<half::bf16>(keys);
                }
            },
            DataType::F16 => {
                if self.target.quantize_keys() {
                    self.decompress_keys_typed::<half::f16>(keys);
                }
            },
            DataType::F32 => {
                if self.target.quantize_keys() {
                    self.decompress_keys_typed::<f32>(keys);
                }
            },
            dtype => panic!("shaped cache does not support dtype {dtype:?}"),
        }
    }

    fn decompress_values(
        &self,
        values: &mut Array<B>,
    ) {
        assert_eq!(values.shape(), &self.shape, "shaped cache value shape mismatch");
        match values.data_type() {
            DataType::BF16 => {
                if self.target.quantize_values() {
                    self.decompress_values_typed::<half::bf16>(values);
                }
            },
            DataType::F16 => {
                if self.target.quantize_values() {
                    self.decompress_values_typed::<half::f16>(values);
                }
            },
            DataType::F32 => {
                if self.target.quantize_values() {
                    self.decompress_values_typed::<f32>(values);
                }
            },
            dtype => panic!("shaped cache does not support dtype {dtype:?}"),
        }
    }

    fn update_rows_from_dense(
        &mut self,
        _context: &B::Context,
        keys: &Array<B>,
        values: &Array<B>,
        source_indices: &[usize],
        destination_indices: &[usize],
    ) {
        assert_eq!(keys.shape(), &self.shape, "shaped cache key shape mismatch");
        assert_eq!(values.shape(), &self.shape, "shaped cache value shape mismatch");
        assert_eq!(keys.data_type(), values.data_type(), "shaped cache KV dtype mismatch");

        match keys.data_type() {
            DataType::BF16 => {
                self.update_rows_from_dense_typed::<half::bf16>(keys, values, source_indices, destination_indices)
            },
            DataType::F16 => {
                self.update_rows_from_dense_typed::<half::f16>(keys, values, source_indices, destination_indices)
            },
            DataType::F32 => {
                self.update_rows_from_dense_typed::<f32>(keys, values, source_indices, destination_indices)
            },
            dtype => panic!("shaped cache does not support dtype {dtype:?}"),
        }
    }

    fn fill_prefix_attention_scores_for_single_decode(
        &self,
        queries: &[f32],
        num_heads: usize,
        prefix_length: usize,
        scores_out: &mut [f32],
    ) -> bool {
        if !self.target.quantize_keys() {
            return false;
        }

        let num_groups = self.shape[0];
        let sequence_length = self.shape[1];
        let head_dim = self.shape[2];
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        assert_eq!(queries.len(), num_heads * head_dim, "shaped cache query shape mismatch");
        assert_eq!(scores_out.len(), num_heads * prefix_length, "shaped cache score shape mismatch");
        assert_eq!(num_heads % num_groups, 0, "shaped cache GQA factor must divide head count");
        assert!(prefix_length <= sequence_length, "shaped cache prefix length out of bounds");

        let gqa_factor = num_heads / num_groups;
        let mut rotated_query = vec![0.0f32; head_dim];

        for head_index in 0..num_heads {
            let query = &queries[head_index * head_dim..(head_index + 1) * head_dim];
            let group_index = head_index / gqa_factor;
            rotate(Self::basis_for_group(&self.key_basis, group_index, head_dim), query, &mut rotated_query);

            for token_index in 0..prefix_length {
                let row_index = group_index * sequence_length + token_index;
                let coefficients = &self.key_coefficients[row_index * self.key_rank..(row_index + 1) * self.key_rank];
                let score = coefficients
                    .iter()
                    .zip(rotated_query.iter())
                    .map(|(coefficient, &query_value)| f32::from(*coefficient) * query_value)
                    .sum::<f32>();
                scores_out[head_index * prefix_length + token_index] = scale * self.key_norms[row_index] * score;
            }
        }

        true
    }

    fn supports_prefix_attention_scores_for_single_decode(&self) -> bool {
        self.target.quantize_keys()
    }

    fn supports_value_row_decoding_for_single_decode(&self) -> bool {
        self.target.quantize_values()
    }

    fn decode_value_row_for_single_decode(
        &self,
        row_index: usize,
        scratch: &mut [f32],
        output: &mut [f32],
    ) {
        assert!(self.target.quantize_values(), "shaped cache value row decoding requires quantized values");
        assert!(row_index < self.shape[0] * self.shape[1], "shaped cache value row index out of bounds");
        assert_eq!(output.len(), self.shape[2], "shaped cache value row shape mismatch");
        self.decode_value_row_into(row_index, scratch, output);
    }

    fn memory_usage_bytes(&self) -> usize {
        let key_storage = if self.target.quantize_keys() {
            self.key_norms.len() * std::mem::size_of::<f32>()
                + self.key_coefficients.len() * std::mem::size_of::<half::bf16>()
        } else {
            0
        };
        let value_storage = if self.target.quantize_values() {
            self.value_norms.len() * std::mem::size_of::<f32>()
                + self.value_coefficients.len() * std::mem::size_of::<half::bf16>()
        } else {
            0
        };
        key_storage + value_storage
    }
}

fn compress_sign_row<T>(
    row: &[T],
    basis: &[f32],
    packed: &mut [u8],
    norm_out: &mut f32,
    normalized: &mut [f32],
    rotated: &mut [f32],
) where
    T: ArrayElement + Copy,
    f32: From<T>,
{
    let norm = row.iter().map(|&value| {
        let value = f32::from(value);
        value * value
    });
    let norm = norm.sum::<f32>().sqrt();
    *norm_out = norm;

    if norm == 0.0 {
        packed.fill(0);
        return;
    }

    for (dst, &src) in normalized.iter_mut().zip(row.iter()) {
        *dst = f32::from(src) / norm;
    }
    rotate(basis, normalized, rotated);
    write_sign_bits(rotated, packed);
}

fn compress_quantized_row<T>(
    row: &[T],
    basis: &[f32],
    layout: &SpectralValueLayout,
    packed: &mut [u8],
    norm_out: &mut f32,
    normalized: &mut [f32],
    rotated: &mut [f32],
) where
    T: ArrayElement + Copy,
    f32: From<T>,
{
    let norm = row.iter().map(|&value| {
        let value = f32::from(value);
        value * value
    });
    let norm = norm.sum::<f32>().sqrt();
    *norm_out = norm;

    if norm == 0.0 {
        packed.fill(0);
        return;
    }

    for (dst, &src) in normalized.iter_mut().zip(row.iter()) {
        *dst = f32::from(src) / norm;
    }
    rotate(basis, normalized, rotated);
    layout.encode_row(rotated, packed);
}

fn compress_truncated_row<T>(
    row: &[T],
    basis: &[f32],
    rank: usize,
    stored: &mut [half::bf16],
    norm_out: &mut f32,
    normalized: &mut [f32],
    rotated: &mut [f32],
) where
    T: ArrayElement + Copy,
    f32: From<T>,
{
    let norm = row.iter().map(|&value| {
        let value = f32::from(value);
        value * value
    });
    let norm = norm.sum::<f32>().sqrt();
    *norm_out = norm;

    if norm == 0.0 {
        stored.fill(half::bf16::ZERO);
        return;
    }

    for (dst, &src) in normalized.iter_mut().zip(row.iter()) {
        *dst = f32::from(src) / norm;
    }
    rotate(basis, normalized, rotated);
    for (dst, &src) in stored.iter_mut().zip(rotated.iter().take(rank)) {
        *dst = half::bf16::from_f32(src);
    }
}

fn write_sign_bits(
    input: &[f32],
    packed: &mut [u8],
) {
    packed.fill(0);
    for (index, &value) in input.iter().enumerate() {
        if value >= 0.0 {
            write_bits(packed, index, 1, 1);
        }
    }
}

fn decode_sign_row(
    packed: &[u8],
    output: &mut [f32],
) {
    let scale = 1.0f32 / (output.len() as f32).sqrt();
    for (index, value) in output.iter_mut().enumerate() {
        *value = if read_bits(packed, index, 1) == 1 {
            scale
        } else {
            -scale
        };
    }
}

fn gaussian_codebook(
    bits: usize,
    head_dim: usize,
) -> Vec<f32> {
    let levels = 1usize << bits;
    let grid_size = 32768usize;
    let min_x = -8.0f32;
    let max_x = 8.0f32;
    let step = (max_x - min_x) / (grid_size - 1) as f32;
    let grid: Vec<f32> = (0..grid_size).map(|index| min_x + step * index as f32).collect();
    let weights: Vec<f32> = grid.iter().map(|&x| gaussian_pdf(x)).collect();

    let mut centroids = (0..levels)
        .map(|index| {
            let fraction = (index as f32 + 0.5) / levels as f32;
            (2.0 * fraction - 1.0) * 3.0
        })
        .collect::<Vec<_>>();

    for _ in 0..64 {
        let mut next = centroids.clone();
        let mut bounds = vec![f32::NEG_INFINITY; levels + 1];
        bounds[levels] = f32::INFINITY;
        for index in 0..levels - 1 {
            bounds[index + 1] = 0.5 * (centroids[index] + centroids[index + 1]);
        }

        for level in 0..levels {
            let mut weighted_sum = 0.0f32;
            let mut total_weight = 0.0f32;
            for (&x, &weight) in grid.iter().zip(weights.iter()) {
                if x >= bounds[level] && x < bounds[level + 1] {
                    weighted_sum += x * weight;
                    total_weight += weight;
                }
            }
            if total_weight > 0.0 {
                next[level] = weighted_sum / total_weight;
            }
        }

        let delta = centroids.iter().zip(next.iter()).map(|(left, right)| (left - right).abs()).fold(0.0f32, f32::max);
        centroids = next;
        if delta < 1e-5 {
            break;
        }
    }

    let scale = (head_dim as f32).sqrt();
    centroids.into_iter().map(|value| value / scale).collect()
}

fn gaussian_pdf(x: f32) -> f32 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

fn rotate(
    matrix: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    let dim = input.len();
    for row in 0..dim {
        let mut sum = 0.0f32;
        for col in 0..dim {
            sum += matrix[row * dim + col] * input[col];
        }
        output[row] = sum;
    }
}

fn inverse_rotate(
    matrix: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    let dim = input.len();
    for col in 0..dim {
        let mut sum = 0.0f32;
        for row in 0..dim {
            sum += matrix[row * dim + col] * input[row];
        }
        output[col] = sum;
    }
}

fn encode_uniform_row(
    input: &[f32],
    codebook: &[f32],
    bits: usize,
    packed: &mut [u8],
) {
    let levels = 1usize << bits;
    let mut bit_offset = 0usize;
    for &value in input {
        let mut best_index = 0usize;
        let mut best_error = f32::INFINITY;
        for index in 0..levels {
            let error = (value - codebook[index]).abs();
            if error < best_error {
                best_error = error;
                best_index = index;
            }
        }
        write_bits(packed, bit_offset, bits, best_index as u32);
        bit_offset += bits;
    }
}

fn encode_multi_regime_row(
    input: &[f32],
    regimes: &[SpectralValueRegime],
    packed: &mut [u8],
) {
    let mut bit_offset = 0usize;
    let mut start_dim = 0usize;
    for regime in regimes {
        let levels = 1usize << regime.bits;
        for &value in &input[start_dim..regime.end_dim] {
            let mut best_index = 0usize;
            let mut best_error = f32::INFINITY;
            for index in 0..levels {
                let error = (value - regime.codebook[index]).abs();
                if error < best_error {
                    best_error = error;
                    best_index = index;
                }
            }
            write_bits(packed, bit_offset, regime.bits, best_index as u32);
            bit_offset += regime.bits;
        }
        start_dim = regime.end_dim;
    }
}

fn decode_uniform_row(
    packed: &[u8],
    codebook: &[f32],
    bits: usize,
    output: &mut [f32],
) {
    let mut bit_offset = 0usize;
    for value in output.iter_mut() {
        let index = read_bits(packed, bit_offset, bits) as usize;
        *value = codebook[index];
        bit_offset += bits;
    }
}

fn decode_multi_regime_row(
    packed: &[u8],
    regimes: &[SpectralValueRegime],
    output: &mut [f32],
) {
    let mut bit_offset = 0usize;
    let mut start_dim = 0usize;
    for regime in regimes {
        for value in &mut output[start_dim..regime.end_dim] {
            let index = read_bits(packed, bit_offset, regime.bits) as usize;
            *value = regime.codebook[index];
            bit_offset += regime.bits;
        }
        start_dim = regime.end_dim;
    }
}

fn decode_truncated_row(
    stored: &[half::bf16],
    output: &mut [f32],
) {
    output.fill(0.0);
    for (dst, &src) in output.iter_mut().zip(stored.iter()) {
        *dst = f32::from(src);
    }
}

fn write_bits(
    packed: &mut [u8],
    bit_offset: usize,
    bits: usize,
    value: u32,
) {
    for bit in 0..bits {
        let absolute = bit_offset + bit;
        let byte_index = absolute / 8;
        let bit_index = absolute % 8;
        let mask = 1u8 << bit_index;
        if ((value >> bit) & 1) == 1 {
            packed[byte_index] |= mask;
        } else {
            packed[byte_index] &= !mask;
        }
    }
}

fn read_bits(
    packed: &[u8],
    bit_offset: usize,
    bits: usize,
) -> u32 {
    let mut value = 0u32;
    for bit in 0..bits {
        let absolute = bit_offset + bit;
        let byte_index = absolute / 8;
        let bit_index = absolute % 8;
        let current = (packed[byte_index] >> bit_index) & 1;
        value |= (current as u32) << bit;
    }
    value
}

fn dot_sign_row(
    packed: &[u8],
    vector: &[f32],
    scale: f32,
) -> f32 {
    vector
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if read_bits(packed, index, 1) == 1 {
                scale * value
            } else {
                -scale * value
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_row_round_trip_has_unit_norm() {
        let mut packed = [0u8; 1];
        write_sign_bits(&[1.0, -2.0, 3.0, -4.0], &mut packed);

        let mut decoded = [0.0f32; 4];
        decode_sign_row(&packed, &mut decoded);

        let norm = decoded.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!(decoded[0] > 0.0);
        assert!(decoded[1] < 0.0);
    }

    #[test]
    fn gaussian_codebook_is_sorted() {
        let codebook = gaussian_codebook(4, 128);
        assert!(codebook.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn decode_truncated_row_zero_fills_tail() {
        let stored = [half::bf16::from_f32(1.5), half::bf16::from_f32(-0.5)];
        let mut output = [7.0f32; 4];
        decode_truncated_row(&stored, &mut output);
        assert_eq!(output, [1.5, -0.5, 0.0, 0.0]);
    }

    #[test]
    fn multi_regime_layout_round_trip_keeps_finite_values() {
        let layout = SpectralValueLayout::new(
            8,
            SpectralValueLayoutSpec::MultiRegime {
                regimes: vec![
                    SpectralValueRegimeSpec {
                        end_dim: 2,
                        bits: 6,
                    },
                    SpectralValueRegimeSpec {
                        end_dim: 5,
                        bits: 3,
                    },
                    SpectralValueRegimeSpec {
                        end_dim: 8,
                        bits: 1,
                    },
                ]
                .into_boxed_slice(),
            },
        );
        let mut packed = vec![0u8; layout.row_bytes()];
        let input = [0.30, -0.21, 0.12, -0.08, 0.05, -0.03, 0.01, -0.01];
        layout.encode_row(&input, &mut packed);

        let mut output = [0.0f32; 8];
        layout.decode_row(&packed, &mut output);

        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output[0].abs() >= output[6].abs());
    }
}
