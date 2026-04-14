use std::{f32::consts::PI, path::Path, sync::Arc};

use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{
    DataType,
    array::Array,
    backends::common::{Backend, Buffer, Context},
    forward_pass::{
        cache_layers::KvCompressionConfig,
        kv_cache_layer::{
            KvCompressionMode, KvCompressor, SingleDecodeValueKernelBuffers, SparseValueConfig, TriAttentionConfig,
        },
        kv_spectral::KvSpectralCalibration,
        kv_spectral_compressor::{
            ShapedCacheCompressor, SpectralQuantCompressor, SpectralQuantTarget, SpectralValueLayoutSpec,
            SpectralValueRegimeSpec,
        },
    },
    utils::env_utils::EnvVar,
};

const TURBOQUANT_SEED: u64 = 0x5eed_cafe_u64;
const TURBOQUANT_QJL_SEED: u64 = 0x71a1_c0de_u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TurboQuantTarget {
    Both,
    Keys,
    Values,
    Exact,
}

impl TurboQuantTarget {
    fn quantize_keys(self) -> bool {
        matches!(self, TurboQuantTarget::Both | TurboQuantTarget::Keys)
    }

    fn quantize_values(self) -> bool {
        matches!(self, TurboQuantTarget::Both | TurboQuantTarget::Values)
    }
}

pub fn env_kv_compression_config<B: Backend>() -> KvCompressionConfig<B> {
    let mode = EnvVar::KvCompression.value();
    if mode.eq_ignore_ascii_case("sparsevalue") || mode.eq_ignore_ascii_case("sparse_value") {
        return KvCompressionConfig {
            mode: KvCompressionMode::SparseValue,
            compress_keys: false,
            compress_values: false,
            factory: None,
            sparse_value: Some(SparseValueConfig {
                recent_window: sparse_value_recent_window(),
                keep_mass: sparse_value_keep_mass(),
                page_size: sparse_value_page_size(),
            }),
            triattention: None,
        };
    }
    if mode.eq_ignore_ascii_case("turboquant") {
        let target = turboquant_target();
        return KvCompressionConfig {
            mode: KvCompressionMode::TurboQuant,
            compress_keys: target.quantize_keys(),
            compress_values: target.quantize_values(),
            factory: Some(Arc::new(move |_, layer_index, shape| {
                Box::new(TurboQuantCompressor::new(layer_index, shape, turboquant_bits(), target))
            })),
            sparse_value: None,
            triattention: None,
        };
    }
    if mode.eq_ignore_ascii_case("shearkv")
        || mode.eq_ignore_ascii_case("shear-kv")
        || mode.eq_ignore_ascii_case("shear_kv")
    {
        return KvCompressionConfig {
            mode: KvCompressionMode::ShearKv,
            compress_keys: false,
            compress_values: true,
            factory: Some(Arc::new(move |context, _layer_index, shape| {
                Box::new(ShearKvCompressor::new(context, shape, shearkv_value_bits()))
            })),
            sparse_value: None,
            triattention: None,
        };
    }
    if mode.eq_ignore_ascii_case("spectralquant") {
        let calibration = Arc::new(KvSpectralCalibration::load(Path::new(&spectral_calibration_path())));
        let target = spectral_target();
        let value_layout = spectral_value_layout();
        return KvCompressionConfig {
            mode: KvCompressionMode::SpectralQuant,
            compress_keys: target.quantize_keys(),
            compress_values: target.quantize_values(),
            factory: Some(Arc::new(move |_, layer_index, shape| {
                Box::new(SpectralQuantCompressor::new(
                    shape,
                    calibration.layer(layer_index),
                    value_layout.clone(),
                    target,
                ))
            })),
            sparse_value: None,
            triattention: None,
        };
    }
    if mode.eq_ignore_ascii_case("sparsevalue_spectral") || mode.eq_ignore_ascii_case("sparsevalue-spectral") {
        let calibration = Arc::new(KvSpectralCalibration::load(Path::new(&spectral_calibration_path())));
        let value_layout = spectral_value_layout();
        return KvCompressionConfig {
            mode: KvCompressionMode::SpectralQuant,
            compress_keys: false,
            compress_values: true,
            factory: Some(Arc::new(move |_, layer_index, shape| {
                Box::new(SpectralQuantCompressor::new(
                    shape,
                    calibration.layer(layer_index),
                    value_layout.clone(),
                    SpectralQuantTarget::Values,
                ))
            })),
            sparse_value: Some(SparseValueConfig {
                recent_window: sparse_value_recent_window(),
                keep_mass: sparse_value_keep_mass(),
                page_size: sparse_value_page_size(),
            }),
            triattention: None,
        };
    }
    if mode.eq_ignore_ascii_case("sparsevalue_turboquant") || mode.eq_ignore_ascii_case("sparsevalue-turboquant") {
        return KvCompressionConfig {
            mode: KvCompressionMode::TurboQuant,
            compress_keys: false,
            compress_values: true,
            factory: Some(Arc::new(move |_, layer_index, shape| {
                Box::new(TurboQuantCompressor::new(layer_index, shape, turboquant_bits(), TurboQuantTarget::Values))
            })),
            sparse_value: Some(SparseValueConfig {
                recent_window: sparse_value_recent_window(),
                keep_mass: sparse_value_keep_mass(),
                page_size: sparse_value_page_size(),
            }),
            triattention: None,
        };
    }
    if mode.eq_ignore_ascii_case("shapedcache") {
        let calibration = Arc::new(KvSpectralCalibration::load(Path::new(&spectral_calibration_path())));
        let target = shaped_target();
        let key_dims = shaped_key_dims();
        let value_dims = shaped_value_dims();
        return KvCompressionConfig {
            mode: KvCompressionMode::ShapedCache,
            compress_keys: target.quantize_keys(),
            compress_values: target.quantize_values(),
            factory: Some(Arc::new(move |_, layer_index, shape| {
                Box::new(ShapedCacheCompressor::new(
                    shape,
                    calibration.layer(layer_index),
                    key_dims,
                    value_dims,
                    target,
                ))
            })),
            sparse_value: None,
            triattention: None,
        };
    }
    if mode.eq_ignore_ascii_case("triattention") {
        return KvCompressionConfig {
            mode: KvCompressionMode::TriAttention,
            compress_keys: false,
            compress_values: false,
            factory: None,
            sparse_value: None,
            triattention: Some(TriAttentionConfig {
                budget: triattention_budget(),
                divide_length: triattention_divide_length(),
            }),
        };
    }
    KvCompressionConfig::disabled()
}

fn spectral_calibration_path() -> String {
    let value = EnvVar::KvSpectralCalibration.value();
    assert!(!value.is_empty(), "UZU_KV_SPECTRAL_CALIBRATION must be set for spectralquant");
    value
}

fn sparse_value_keep_mass() -> f32 {
    let value = EnvVar::KvSparseValueKeepMass.value();
    if value.is_empty() {
        return 0.9;
    }
    let keep_mass = value.parse::<f32>().expect("UZU_KV_SPARSE_VALUE_KEEP_MASS must be a float");
    assert!(
        keep_mass.is_finite() && keep_mass > 0.0 && keep_mass <= 1.0,
        "UZU_KV_SPARSE_VALUE_KEEP_MASS must be in (0, 1]"
    );
    keep_mass
}

fn sparse_value_recent_window() -> usize {
    let value = EnvVar::KvSparseValueRecentWindow.value();
    if value.is_empty() {
        return 256;
    }
    value.parse::<usize>().expect("UZU_KV_SPARSE_VALUE_RECENT_WINDOW must be an integer")
}

fn sparse_value_page_size() -> usize {
    let value = EnvVar::KvSparseValuePageSize.value();
    if value.is_empty() {
        return 32;
    }
    let page_size = value.parse::<usize>().expect("UZU_KV_SPARSE_VALUE_PAGE_SIZE must be an integer");
    assert!(page_size > 0, "UZU_KV_SPARSE_VALUE_PAGE_SIZE must be positive");
    page_size
}

fn spectral_value_bits() -> usize {
    let value = EnvVar::KvSpectralValueBits.value();
    if value.is_empty() {
        return 4;
    }
    let bits = value.parse::<usize>().expect("UZU_KV_SPECTRAL_VALUE_BITS must be an integer");
    assert!((1..=8).contains(&bits), "UZU_KV_SPECTRAL_VALUE_BITS must be in [1, 8]");
    bits
}

fn spectral_value_layout() -> SpectralValueLayoutSpec {
    let value = EnvVar::KvSpectralValueLayout.value();
    if value.is_empty() {
        return SpectralValueLayoutSpec::Uniform {
            bits: spectral_value_bits(),
        };
    }

    let regimes = value
        .split(',')
        .map(|entry| {
            let (end_dim, bits) =
                entry.split_once(':').expect("UZU_KV_SPECTRAL_VALUE_LAYOUT entries must look like end_dim:bits");
            SpectralValueRegimeSpec {
                end_dim: end_dim
                    .trim()
                    .parse::<usize>()
                    .expect("UZU_KV_SPECTRAL_VALUE_LAYOUT end dims must be integers"),
                bits: bits.trim().parse::<usize>().expect("UZU_KV_SPECTRAL_VALUE_LAYOUT bits must be integers"),
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();

    SpectralValueLayoutSpec::MultiRegime {
        regimes,
    }
}

fn spectral_target() -> SpectralQuantTarget {
    let value = EnvVar::KvSpectralTarget.value();
    if value.is_empty() {
        return SpectralQuantTarget::Both;
    }
    match value.to_ascii_lowercase().as_str() {
        "both" => SpectralQuantTarget::Both,
        "keys" | "key" => SpectralQuantTarget::Keys,
        "values" | "value" => SpectralQuantTarget::Values,
        "exact" | "none" => SpectralQuantTarget::Exact,
        _ => panic!("UZU_KV_SPECTRAL_TARGET must be one of: both, keys, values, exact"),
    }
}

fn shaped_key_dims() -> usize {
    let value = EnvVar::KvShapedKeyDims.value();
    assert!(!value.is_empty(), "UZU_KV_SHAPED_KEY_DIMS must be set for shapedcache");
    value.parse::<usize>().expect("UZU_KV_SHAPED_KEY_DIMS must be an integer")
}

fn shaped_value_dims() -> usize {
    let value = EnvVar::KvShapedValueDims.value();
    assert!(!value.is_empty(), "UZU_KV_SHAPED_VALUE_DIMS must be set for shapedcache");
    value.parse::<usize>().expect("UZU_KV_SHAPED_VALUE_DIMS must be an integer")
}

fn shaped_target() -> SpectralQuantTarget {
    let value = EnvVar::KvShapedTarget.value();
    if value.is_empty() {
        return SpectralQuantTarget::Both;
    }
    match value.to_ascii_lowercase().as_str() {
        "both" => SpectralQuantTarget::Both,
        "keys" | "key" => SpectralQuantTarget::Keys,
        "values" | "value" => SpectralQuantTarget::Values,
        "exact" | "none" => SpectralQuantTarget::Exact,
        _ => panic!("UZU_KV_SHAPED_TARGET must be one of: both, keys, values, exact"),
    }
}

fn turboquant_bits() -> usize {
    let value = EnvVar::KvTurboQuantBits.value();
    if value.is_empty() {
        return 4;
    }
    let bits = value.parse::<usize>().expect("UZU_KV_TURBOQUANT_BITS must be an integer");
    assert!((1..=8).contains(&bits), "UZU_KV_TURBOQUANT_BITS must be in [1, 8]");
    bits
}

fn shearkv_value_bits() -> usize {
    let value = EnvVar::KvShearKvValueBits.value();
    if value.is_empty() {
        return 8;
    }
    let bits = value.parse::<usize>().expect("UZU_KV_SHEARKV_VALUE_BITS must be an integer");
    assert!(matches!(bits, 4 | 8), "UZU_KV_SHEARKV_VALUE_BITS must be 4 or 8");
    bits
}

fn turboquant_target() -> TurboQuantTarget {
    let value = EnvVar::KvTurboQuantTarget.value();
    if value.is_empty() {
        return TurboQuantTarget::Both;
    }
    match value.to_ascii_lowercase().as_str() {
        "both" => TurboQuantTarget::Both,
        "keys" | "key" => TurboQuantTarget::Keys,
        "values" | "value" => TurboQuantTarget::Values,
        "exact" | "none" => TurboQuantTarget::Exact,
        _ => panic!("UZU_KV_TURBOQUANT_TARGET must be one of: both, keys, values, exact"),
    }
}

fn triattention_budget() -> usize {
    let value = EnvVar::KvTriAttentionBudget.value();
    if value.is_empty() {
        return 2048;
    }
    let budget = value.parse::<usize>().expect("UZU_KV_TRIATTENTION_BUDGET must be an integer");
    assert!(budget > 0, "UZU_KV_TRIATTENTION_BUDGET must be positive");
    budget
}

fn triattention_divide_length() -> usize {
    let value = EnvVar::KvTriAttentionDivideLength.value();
    if value.is_empty() {
        return 128;
    }
    let divide_length = value.parse::<usize>().expect("UZU_KV_TRIATTENTION_DIVIDE_LENGTH must be an integer");
    assert!(divide_length > 0, "UZU_KV_TRIATTENTION_DIVIDE_LENGTH must be positive");
    divide_length
}

struct ShearKvCompressor<B: Backend> {
    shape: [usize; 3],
    bits: usize,
    row_bytes: usize,
    codes: B::Buffer,
    scales: B::Buffer,
    biases: B::Buffer,
}

impl<B: Backend> ShearKvCompressor<B> {
    fn new(
        context: &B::Context,
        shape: [usize; 3],
        bits: usize,
    ) -> Self {
        assert!(matches!(bits, 4 | 8), "ShearKV only supports 4-bit or 8-bit values");
        if bits == 4 {
            assert!(shape[2] % 2 == 0, "ShearKV 4-bit rows require an even head dimension");
        }
        let row_count = shape[0] * shape[1];
        let row_bytes = if bits == 8 {
            shape[2]
        } else {
            shape[2] / 2
        };
        let mut codes = context.create_buffer(row_count * row_bytes).expect("Failed to create ShearKV codes buffer");
        codes.set_label(Some("shearkv_codes"));
        let mut scales = context
            .create_buffer(row_count * std::mem::size_of::<f32>())
            .expect("Failed to create ShearKV scales buffer");
        scales.set_label(Some("shearkv_scales"));
        let mut biases = context
            .create_buffer(row_count * std::mem::size_of::<f32>())
            .expect("Failed to create ShearKV biases buffer");
        biases.set_label(Some("shearkv_biases"));
        Self {
            shape,
            bits,
            row_bytes,
            codes,
            scales,
            biases,
        }
    }

    fn row_count(&self) -> usize {
        self.shape[0] * self.shape[1]
    }

    fn codes_row(
        &self,
        row_index: usize,
    ) -> &[u8] {
        &buffer_as_bytes::<B>(&self.codes)[row_index * self.row_bytes..(row_index + 1) * self.row_bytes]
    }

    fn encode_values<T>(
        &mut self,
        values: &Array<B>,
    ) where
        T: crate::ArrayElement + Copy,
        f32: From<T>,
    {
        assert_eq!(values.shape(), &self.shape, "ShearKV value shape mismatch");
        let head_dim = self.shape[2];
        let codes = buffer_as_bytes_mut::<B>(&mut self.codes);
        let scales = buffer_as_slice_mut::<f32, B>(&mut self.scales);
        let biases = buffer_as_slice_mut::<f32, B>(&mut self.biases);
        for (row_index, row) in values.as_slice::<T>().chunks_exact(head_dim).enumerate() {
            let (row_scale, row_bias) = shear_quantize_row(
                row,
                self.bits,
                &mut codes[row_index * self.row_bytes..(row_index + 1) * self.row_bytes],
            );
            scales[row_index] = row_scale;
            biases[row_index] = row_bias;
        }
    }

    fn update_values<T>(
        &mut self,
        values: &Array<B>,
        source_indices: &[usize],
        destination_indices: &[usize],
    ) where
        T: crate::ArrayElement + Copy,
        f32: From<T>,
    {
        let group_count = self.shape[0];
        let sequence_length = self.shape[1];
        let head_dim = self.shape[2];
        let value_rows = values.as_slice::<T>();
        let codes = buffer_as_bytes_mut::<B>(&mut self.codes);
        let scales = buffer_as_slice_mut::<f32, B>(&mut self.scales);
        let biases = buffer_as_slice_mut::<f32, B>(&mut self.biases);
        for (&source_index, &destination_index) in source_indices.iter().zip(destination_indices.iter()) {
            assert!(source_index < sequence_length, "ShearKV source row out of bounds");
            assert!(destination_index < sequence_length, "ShearKV destination row out of bounds");
            for group_index in 0..group_count {
                let source_row_index = group_index * sequence_length + source_index;
                let destination_row_index = group_index * sequence_length + destination_index;
                let row = &value_rows[source_row_index * head_dim..(source_row_index + 1) * head_dim];
                let (scale, bias) = shear_quantize_row(
                    row,
                    self.bits,
                    &mut codes[destination_row_index * self.row_bytes..(destination_row_index + 1) * self.row_bytes],
                );
                scales[destination_row_index] = scale;
                biases[destination_row_index] = bias;
            }
        }
    }

    fn decode_row_into(
        &self,
        row_index: usize,
        output: &mut [f32],
    ) {
        assert!(row_index < self.row_count(), "ShearKV row index out of bounds");
        assert_eq!(output.len(), self.shape[2], "ShearKV row shape mismatch");
        let scales = buffer_as_slice::<f32, B>(&self.scales);
        let biases = buffer_as_slice::<f32, B>(&self.biases);
        shear_decode_row(self.codes_row(row_index), self.bits, scales[row_index], biases[row_index], output);
    }

    fn decode_rows_into(
        &self,
        row_index: usize,
        row_count: usize,
        output: &mut [f32],
    ) {
        if row_count == 0 {
            assert!(output.is_empty(), "ShearKV decode output must be empty when row_count is zero");
            return;
        }
        let head_dim = self.shape[2];
        assert_eq!(output.len(), row_count * head_dim, "ShearKV row-range output shape mismatch");
        let scales = buffer_as_slice::<f32, B>(&self.scales);
        let biases = buffer_as_slice::<f32, B>(&self.biases);
        for (offset, row) in output.chunks_exact_mut(head_dim).enumerate() {
            let current_row = row_index + offset;
            assert!(current_row < self.row_count(), "ShearKV row range out of bounds");
            shear_decode_row(self.codes_row(current_row), self.bits, scales[current_row], biases[current_row], row);
        }
    }

    fn notify_gpu_buffers_modified(&self) {
        self.codes.did_modify_range(0..self.codes.length());
        self.scales.did_modify_range(0..self.scales.length());
        self.biases.did_modify_range(0..self.biases.length());
    }
}

impl<B: Backend> KvCompressor<B> for ShearKvCompressor<B> {
    fn compress(
        &mut self,
        _keys: &Array<B>,
        values: &Array<B>,
    ) {
        match values.data_type() {
            DataType::BF16 => self.encode_values::<half::bf16>(values),
            DataType::F16 => self.encode_values::<half::f16>(values),
            DataType::F32 => self.encode_values::<f32>(values),
            dtype => panic!("ShearKV does not support KV dtype {dtype:?}"),
        }
        self.notify_gpu_buffers_modified();
    }

    fn decompress(
        &self,
        keys: &mut Array<B>,
        values: &mut Array<B>,
    ) {
        let _ = keys;
        self.decompress_values(values);
    }

    fn decompress_values(
        &self,
        values: &mut Array<B>,
    ) {
        assert_eq!(values.shape(), &self.shape, "ShearKV value shape mismatch");
        let head_dim = self.shape[2];
        let row_count = self.row_count();
        let mut decoded = vec![0.0f32; head_dim];
        match values.data_type() {
            DataType::BF16 => {
                let output = values.as_slice_mut::<half::bf16>();
                for row_index in 0..row_count {
                    self.decode_row_into(row_index, &mut decoded);
                    for (dst, &src) in
                        output[row_index * head_dim..(row_index + 1) * head_dim].iter_mut().zip(decoded.iter())
                    {
                        *dst = half::bf16::from_f32(src);
                    }
                }
            },
            DataType::F16 => {
                let output = values.as_slice_mut::<half::f16>();
                for row_index in 0..row_count {
                    self.decode_row_into(row_index, &mut decoded);
                    for (dst, &src) in
                        output[row_index * head_dim..(row_index + 1) * head_dim].iter_mut().zip(decoded.iter())
                    {
                        *dst = half::f16::from_f32(src);
                    }
                }
            },
            DataType::F32 => {
                let output = values.as_slice_mut::<f32>();
                for row_index in 0..row_count {
                    self.decode_row_into(row_index, &mut output[row_index * head_dim..(row_index + 1) * head_dim]);
                }
            },
            dtype => panic!("ShearKV does not support KV dtype {dtype:?}"),
        }
    }

    fn update_rows_from_dense(
        &mut self,
        _context: &B::Context,
        _keys: &Array<B>,
        values: &Array<B>,
        source_indices: &[usize],
        destination_indices: &[usize],
    ) {
        assert_eq!(source_indices.len(), destination_indices.len(), "ShearKV row update index count mismatch");
        match values.data_type() {
            DataType::BF16 => self.update_values::<half::bf16>(values, source_indices, destination_indices),
            DataType::F16 => self.update_values::<half::f16>(values, source_indices, destination_indices),
            DataType::F32 => self.update_values::<f32>(values, source_indices, destination_indices),
            dtype => panic!("ShearKV does not support KV dtype {dtype:?}"),
        }
        self.notify_gpu_buffers_modified();
    }

    fn supports_value_row_decoding_for_single_decode(&self) -> bool {
        true
    }

    fn value_kernel_buffers_for_single_decode(&self) -> Option<SingleDecodeValueKernelBuffers<'_, B>> {
        Some(SingleDecodeValueKernelBuffers {
            codes: &self.codes,
            scales: &self.scales,
            biases: &self.biases,
            bits: self.bits,
            row_bytes: self.row_bytes,
        })
    }

    fn decode_value_row_for_single_decode(
        &self,
        row_index: usize,
        _scratch: &mut [f32],
        output: &mut [f32],
    ) {
        self.decode_row_into(row_index, output);
    }

    fn decode_value_rows_for_single_decode(
        &self,
        row_index: usize,
        row_count: usize,
        output: &mut [f32],
    ) {
        self.decode_rows_into(row_index, row_count, output);
    }

    fn memory_usage_bytes(&self) -> usize {
        self.codes.length() + self.scales.length() + self.biases.length()
    }
}

fn buffer_as_bytes<B: Backend>(buffer: &B::Buffer) -> &[u8] {
    unsafe { std::slice::from_raw_parts(buffer.cpu_ptr().as_ptr() as *const u8, buffer.length()) }
}

fn buffer_as_bytes_mut<B: Backend>(buffer: &mut B::Buffer) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(buffer.cpu_ptr().as_ptr() as *mut u8, buffer.length()) }
}

fn buffer_as_slice<T: crate::ArrayElement, B: Backend>(buffer: &B::Buffer) -> &[T] {
    bytemuck::cast_slice(buffer_as_bytes::<B>(buffer))
}

fn buffer_as_slice_mut<T: crate::ArrayElement, B: Backend>(buffer: &mut B::Buffer) -> &mut [T] {
    bytemuck::cast_slice_mut(buffer_as_bytes_mut::<B>(buffer))
}

fn shear_quantize_row<T>(
    row: &[T],
    bits: usize,
    packed: &mut [u8],
) -> (f32, f32)
where
    T: crate::ArrayElement + Copy,
    f32: From<T>,
{
    let (row_min, row_max) = row.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min_value, max_value), &value| {
        let value = f32::from(value);
        (min_value.min(value), max_value.max(value))
    });
    let levels = ((1usize << bits) - 1) as f32;
    let scale = if row_max > row_min {
        (row_max - row_min) / levels
    } else {
        0.0
    };
    let bias = row_min;
    match bits {
        8 => {
            for (dst, &src) in packed.iter_mut().zip(row.iter()) {
                let quantized = if scale == 0.0 {
                    0
                } else {
                    (((f32::from(src) - bias) / scale).round()).clamp(0.0, levels) as u8
                };
                *dst = quantized;
            }
        },
        4 => {
            packed.fill(0);
            for (dim, &src) in row.iter().enumerate() {
                let quantized = if scale == 0.0 {
                    0
                } else {
                    (((f32::from(src) - bias) / scale).round()).clamp(0.0, levels) as u8
                };
                let slot = &mut packed[dim / 2];
                if dim % 2 == 0 {
                    *slot = (*slot & 0xF0) | (quantized & 0x0F);
                } else {
                    *slot = (*slot & 0x0F) | ((quantized & 0x0F) << 4);
                }
            }
        },
        _ => panic!("ShearKV only supports 4-bit or 8-bit values"),
    }
    (scale, bias)
}

fn shear_decode_row(
    packed: &[u8],
    bits: usize,
    scale: f32,
    bias: f32,
    output: &mut [f32],
) {
    match bits {
        8 => {
            for (dst, &src) in output.iter_mut().zip(packed.iter()) {
                *dst = bias + scale * src as f32;
            }
        },
        4 => {
            for (dim, dst) in output.iter_mut().enumerate() {
                let byte = packed[dim / 2];
                let quantized = if dim % 2 == 0 {
                    byte & 0x0F
                } else {
                    byte >> 4
                };
                *dst = bias + scale * quantized as f32;
            }
        },
        _ => panic!("ShearKV only supports 4-bit or 8-bit values"),
    }
}

struct TurboQuantCompressor {
    shape: [usize; 3],
    target: TurboQuantTarget,
    key_bits: usize,
    value_bits: usize,
    key_row_bytes: usize,
    value_row_bytes: usize,
    key_residual_row_bytes: usize,
    key_codebook: Box<[f32]>,
    value_codebook: Box<[f32]>,
    rotation: Box<[f32]>,
    qjl_projection: Box<[f32]>,
    key_codes: Vec<u8>,
    value_codes: Vec<u8>,
    key_norms: Vec<f32>,
    value_norms: Vec<f32>,
    key_residual_signs: Vec<u8>,
    key_residual_norms: Vec<f32>,
    dense_row_bytes: usize,
    dense_data_type: Option<DataType>,
}

impl TurboQuantCompressor {
    fn new(
        layer_index: usize,
        shape: [usize; 3],
        bits: usize,
        target: TurboQuantTarget,
    ) -> Self {
        if target.quantize_keys() {
            assert!(bits >= 2, "TurboQuant key quantization requires at least 2 total bits");
        }
        let row_count = shape[0] * shape[1];
        let key_bits = if target.quantize_keys() {
            bits - 1
        } else {
            0
        };
        let value_bits = if target.quantize_values() {
            bits
        } else {
            0
        };
        let key_row_bytes = if target.quantize_keys() {
            (shape[2] * key_bits).div_ceil(8)
        } else {
            0
        };
        let value_row_bytes = if target.quantize_values() {
            (shape[2] * value_bits).div_ceil(8)
        } else {
            0
        };
        let key_residual_row_bytes = if target.quantize_keys() {
            shape[2].div_ceil(8)
        } else {
            0
        };
        Self {
            shape,
            target,
            key_bits,
            value_bits,
            key_row_bytes,
            value_row_bytes,
            key_residual_row_bytes,
            key_codebook: if target.quantize_keys() {
                gaussian_codebook(key_bits, shape[2]).into_boxed_slice()
            } else {
                Vec::new().into_boxed_slice()
            },
            value_codebook: if target.quantize_values() {
                gaussian_codebook(value_bits, shape[2]).into_boxed_slice()
            } else {
                Vec::new().into_boxed_slice()
            },
            rotation: random_orthogonal(shape[2], TURBOQUANT_SEED ^ layer_index as u64).into_boxed_slice(),
            qjl_projection: scaled_random_orthogonal(shape[2], TURBOQUANT_QJL_SEED ^ layer_index as u64)
                .into_boxed_slice(),
            key_codes: vec![0; row_count * key_row_bytes],
            value_codes: vec![0; row_count * value_row_bytes],
            key_norms: vec![0.0; row_count],
            value_norms: vec![0.0; row_count],
            key_residual_signs: vec![0; row_count * key_residual_row_bytes],
            key_residual_norms: vec![0.0; row_count],
            dense_row_bytes: 0,
            dense_data_type: None,
        }
    }

    fn ensure_dense_storage(
        &mut self,
        data_type: DataType,
    ) {
        if let Some(existing) = self.dense_data_type {
            assert_eq!(existing, data_type, "TurboQuant dense storage dtype mismatch");
            return;
        }

        let row_count = self.shape[0] * self.shape[1];
        self.dense_row_bytes = self.shape[2] * data_type.size_in_bytes();
        self.dense_data_type = Some(data_type);
        let _ = row_count;
    }

    fn compress_array<T>(
        shape: [usize; 3],
        row_bytes: usize,
        rotation: &[f32],
        codebook: &[f32],
        bits: usize,
        array: &Array<impl Backend>,
        codes: &mut [u8],
        norms: &mut [f32],
    ) where
        T: crate::ArrayElement + Copy,
        f32: From<T>,
    {
        let head_dim = shape[2];
        let mut normalized = vec![0.0f32; head_dim];
        let mut rotated = vec![0.0f32; head_dim];
        for (row_index, row) in array.as_slice::<T>().chunks_exact(head_dim).enumerate() {
            let norm = row.iter().map(|&value| {
                let value = f32::from(value);
                value * value
            });
            let norm = norm.sum::<f32>().sqrt();
            norms[row_index] = norm;

            if norm == 0.0 {
                codes[row_index * row_bytes..(row_index + 1) * row_bytes].fill(0);
                continue;
            }

            for (dst, &src) in normalized.iter_mut().zip(row.iter()) {
                *dst = f32::from(src) / norm;
            }
            rotate(rotation, &normalized, &mut rotated);
            encode_row(&rotated, codebook, bits, &mut codes[row_index * row_bytes..(row_index + 1) * row_bytes]);
        }
    }

    fn compress_row<T>(
        row: &[T],
        rotation: &[f32],
        codebook: &[f32],
        bits: usize,
        packed: &mut [u8],
        norm_out: &mut f32,
        normalized: &mut [f32],
        rotated: &mut [f32],
    ) where
        T: crate::ArrayElement + Copy,
        f32: From<T>,
    {
        let norm = row
            .iter()
            .map(|&value| {
                let value = f32::from(value);
                value * value
            })
            .sum::<f32>()
            .sqrt();
        *norm_out = norm;

        if norm == 0.0 {
            packed.fill(0);
            return;
        }

        for (dst, &src) in normalized.iter_mut().zip(row.iter()) {
            *dst = f32::from(src) / norm;
        }
        rotate(rotation, normalized, rotated);
        encode_row(rotated, codebook, bits, packed);
    }

    fn compress_key_array<T>(
        shape: [usize; 3],
        key_row_bytes: usize,
        key_residual_row_bytes: usize,
        rotation: &[f32],
        key_codebook: &[f32],
        key_bits: usize,
        qjl_projection: &[f32],
        array: &Array<impl Backend>,
        codes: &mut [u8],
        norms: &mut [f32],
        residual_signs: &mut [u8],
        residual_norms: &mut [f32],
    ) where
        T: crate::ArrayElement + Copy,
        f32: From<T>,
    {
        let head_dim = shape[2];
        let mut normalized = vec![0.0f32; head_dim];
        let mut rotated = vec![0.0f32; head_dim];
        let mut approx_rotated = vec![0.0f32; head_dim];
        let mut approx_dense = vec![0.0f32; head_dim];
        let mut residual = vec![0.0f32; head_dim];

        for (row_index, row) in array.as_slice::<T>().chunks_exact(head_dim).enumerate() {
            let code_range = row_index * key_row_bytes..(row_index + 1) * key_row_bytes;
            let residual_range = row_index * key_residual_row_bytes..(row_index + 1) * key_residual_row_bytes;
            Self::compress_key_row(
                row,
                rotation,
                key_codebook,
                key_bits,
                qjl_projection,
                &mut codes[code_range],
                &mut norms[row_index],
                &mut residual_signs[residual_range],
                &mut residual_norms[row_index],
                &mut normalized,
                &mut rotated,
                &mut approx_rotated,
                &mut approx_dense,
                &mut residual,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compress_key_row<T>(
        row: &[T],
        rotation: &[f32],
        key_codebook: &[f32],
        key_bits: usize,
        qjl_projection: &[f32],
        packed: &mut [u8],
        norm_out: &mut f32,
        residual_packed: &mut [u8],
        residual_norm_out: &mut f32,
        normalized: &mut [f32],
        rotated: &mut [f32],
        approx_rotated: &mut [f32],
        approx_dense: &mut [f32],
        residual: &mut [f32],
    ) where
        T: crate::ArrayElement + Copy,
        f32: From<T>,
    {
        Self::compress_row(row, rotation, key_codebook, key_bits, packed, norm_out, normalized, rotated);
        if *norm_out == 0.0 {
            residual_packed.fill(0);
            *residual_norm_out = 0.0;
            return;
        }

        decode_row(packed, key_codebook, key_bits, approx_rotated);
        inverse_rotate(rotation, approx_rotated, approx_dense);

        for (index, approx_value) in approx_dense.iter_mut().enumerate() {
            *approx_value *= *norm_out;
            residual[index] = f32::from(row[index]) - *approx_value;
        }

        *residual_norm_out = residual.iter().map(|value| value * value).sum::<f32>().sqrt();
        fill_sign_bits(qjl_projection, residual, residual_packed);
    }

    fn update_rows_from_dense_typed<T>(
        &mut self,
        keys: &Array<impl Backend>,
        values: &Array<impl Backend>,
        source_indices: &[usize],
        destination_indices: &[usize],
    ) where
        T: crate::ArrayElement + Copy,
        f32: From<T>,
    {
        assert_eq!(source_indices.len(), destination_indices.len(), "TurboQuant row update index mismatch");

        if !self.target.quantize_keys() && !self.target.quantize_values() {
            return;
        }

        let group_count = self.shape[0];
        let sequence_length = self.shape[1];
        let head_dim = self.shape[2];
        let key_rows = keys.as_slice::<T>();
        let value_rows = values.as_slice::<T>();
        let mut normalized = vec![0.0f32; head_dim];
        let mut rotated = vec![0.0f32; head_dim];
        let mut approx_rotated = vec![0.0f32; head_dim];
        let mut approx_dense = vec![0.0f32; head_dim];
        let mut residual = vec![0.0f32; head_dim];

        for (&source_index, &destination_index) in source_indices.iter().zip(destination_indices.iter()) {
            assert!(source_index < sequence_length, "TurboQuant source row out of bounds");
            assert!(destination_index < sequence_length, "TurboQuant destination row out of bounds");
            for group_index in 0..group_count {
                let source_row_index = group_index * sequence_length + source_index;
                let destination_row_index = group_index * sequence_length + destination_index;
                let row_start = source_row_index * head_dim;
                let row_end = row_start + head_dim;

                if self.target.quantize_keys() {
                    Self::compress_key_row(
                        &key_rows[row_start..row_end],
                        &self.rotation,
                        &self.key_codebook,
                        self.key_bits,
                        &self.qjl_projection,
                        &mut self.key_codes[destination_row_index * self.key_row_bytes
                            ..(destination_row_index + 1) * self.key_row_bytes],
                        &mut self.key_norms[destination_row_index],
                        &mut self.key_residual_signs[destination_row_index * self.key_residual_row_bytes
                            ..(destination_row_index + 1) * self.key_residual_row_bytes],
                        &mut self.key_residual_norms[destination_row_index],
                        &mut normalized,
                        &mut rotated,
                        &mut approx_rotated,
                        &mut approx_dense,
                        &mut residual,
                    );
                }
                if self.target.quantize_values() {
                    Self::compress_row(
                        &value_rows[row_start..row_end],
                        &self.rotation,
                        &self.value_codebook,
                        self.value_bits,
                        &mut self.value_codes[destination_row_index * self.value_row_bytes
                            ..(destination_row_index + 1) * self.value_row_bytes],
                        &mut self.value_norms[destination_row_index],
                        &mut normalized,
                        &mut rotated,
                    );
                }
            }
        }
    }

    fn decompress_array_f32(
        &self,
        array: &mut Array<impl Backend>,
        codes: &[u8],
        norms: &[f32],
        row_bytes: usize,
        codebook: &[f32],
        bits: usize,
    ) {
        let head_dim = self.shape[2];
        let mut rotated = vec![0.0f32; head_dim];
        let mut restored = vec![0.0f32; head_dim];
        for (row_index, row) in array.as_slice_mut::<f32>().chunks_exact_mut(head_dim).enumerate() {
            let norm = norms[row_index];
            if norm == 0.0 {
                for value in row.iter_mut() {
                    *value = 0.0;
                }
                continue;
            }
            decode_row(&codes[row_index * row_bytes..(row_index + 1) * row_bytes], codebook, bits, &mut rotated);
            inverse_rotate(&self.rotation, &rotated, &mut restored);
            for (dst, &src) in row.iter_mut().zip(restored.iter()) {
                *dst = src * norm;
            }
        }
    }

    fn decompress_array_f16(
        &self,
        array: &mut Array<impl Backend>,
        codes: &[u8],
        norms: &[f32],
        row_bytes: usize,
        codebook: &[f32],
        bits: usize,
    ) {
        let head_dim = self.shape[2];
        let mut rotated = vec![0.0f32; head_dim];
        let mut restored = vec![0.0f32; head_dim];
        for (row_index, row) in array.as_slice_mut::<half::f16>().chunks_exact_mut(head_dim).enumerate() {
            let norm = norms[row_index];
            if norm == 0.0 {
                for value in row.iter_mut() {
                    *value = half::f16::from_f32(0.0);
                }
                continue;
            }
            decode_row(&codes[row_index * row_bytes..(row_index + 1) * row_bytes], codebook, bits, &mut rotated);
            inverse_rotate(&self.rotation, &rotated, &mut restored);
            for (dst, &src) in row.iter_mut().zip(restored.iter()) {
                *dst = half::f16::from_f32(src * norm);
            }
        }
    }

    fn decompress_array_bf16(
        &self,
        array: &mut Array<impl Backend>,
        codes: &[u8],
        norms: &[f32],
        row_bytes: usize,
        codebook: &[f32],
        bits: usize,
    ) {
        let head_dim = self.shape[2];
        let mut rotated = vec![0.0f32; head_dim];
        let mut restored = vec![0.0f32; head_dim];
        for (row_index, row) in array.as_slice_mut::<half::bf16>().chunks_exact_mut(head_dim).enumerate() {
            let norm = norms[row_index];
            if norm == 0.0 {
                for value in row.iter_mut() {
                    *value = half::bf16::from_f32(0.0);
                }
                continue;
            }
            decode_row(&codes[row_index * row_bytes..(row_index + 1) * row_bytes], codebook, bits, &mut rotated);
            inverse_rotate(&self.rotation, &rotated, &mut restored);
            for (dst, &src) in row.iter_mut().zip(restored.iter()) {
                *dst = half::bf16::from_f32(src * norm);
            }
        }
    }
}

impl<B: Backend> KvCompressor<B> for TurboQuantCompressor {
    fn compress(
        &mut self,
        keys: &Array<B>,
        values: &Array<B>,
    ) {
        assert_eq!(keys.shape(), &self.shape, "TurboQuant key shape mismatch");
        assert_eq!(values.shape(), &self.shape, "TurboQuant value shape mismatch");
        assert_eq!(keys.data_type(), values.data_type(), "TurboQuant KV dtype mismatch");
        self.ensure_dense_storage(keys.data_type());

        match keys.data_type() {
            DataType::BF16 => {
                if self.target.quantize_keys() {
                    Self::compress_key_array::<half::bf16>(
                        self.shape,
                        self.key_row_bytes,
                        self.key_residual_row_bytes,
                        &self.rotation,
                        &self.key_codebook,
                        self.key_bits,
                        &self.qjl_projection,
                        keys,
                        &mut self.key_codes,
                        &mut self.key_norms,
                        &mut self.key_residual_signs,
                        &mut self.key_residual_norms,
                    );
                }
                if self.target.quantize_values() {
                    Self::compress_array::<half::bf16>(
                        self.shape,
                        self.value_row_bytes,
                        &self.rotation,
                        &self.value_codebook,
                        self.value_bits,
                        values,
                        &mut self.value_codes,
                        &mut self.value_norms,
                    );
                }
            },
            DataType::F16 => {
                if self.target.quantize_keys() {
                    Self::compress_key_array::<half::f16>(
                        self.shape,
                        self.key_row_bytes,
                        self.key_residual_row_bytes,
                        &self.rotation,
                        &self.key_codebook,
                        self.key_bits,
                        &self.qjl_projection,
                        keys,
                        &mut self.key_codes,
                        &mut self.key_norms,
                        &mut self.key_residual_signs,
                        &mut self.key_residual_norms,
                    );
                }
                if self.target.quantize_values() {
                    Self::compress_array::<half::f16>(
                        self.shape,
                        self.value_row_bytes,
                        &self.rotation,
                        &self.value_codebook,
                        self.value_bits,
                        values,
                        &mut self.value_codes,
                        &mut self.value_norms,
                    );
                }
            },
            DataType::F32 => {
                if self.target.quantize_keys() {
                    Self::compress_key_array::<f32>(
                        self.shape,
                        self.key_row_bytes,
                        self.key_residual_row_bytes,
                        &self.rotation,
                        &self.key_codebook,
                        self.key_bits,
                        &self.qjl_projection,
                        keys,
                        &mut self.key_codes,
                        &mut self.key_norms,
                        &mut self.key_residual_signs,
                        &mut self.key_residual_norms,
                    );
                }
                if self.target.quantize_values() {
                    Self::compress_array::<f32>(
                        self.shape,
                        self.value_row_bytes,
                        &self.rotation,
                        &self.value_codebook,
                        self.value_bits,
                        values,
                        &mut self.value_codes,
                        &mut self.value_norms,
                    );
                }
            },
            dtype => panic!("TurboQuant does not support KV dtype {dtype:?}"),
        }
    }

    fn decompress(
        &self,
        keys: &mut Array<B>,
        values: &mut Array<B>,
    ) {
        assert_eq!(keys.shape(), &self.shape, "TurboQuant key shape mismatch");
        assert_eq!(values.shape(), &self.shape, "TurboQuant value shape mismatch");
        assert_eq!(keys.data_type(), values.data_type(), "TurboQuant KV dtype mismatch");

        match keys.data_type() {
            DataType::BF16 => {
                if self.target.quantize_keys() {
                    self.decompress_array_bf16(
                        keys,
                        &self.key_codes,
                        &self.key_norms,
                        self.key_row_bytes,
                        &self.key_codebook,
                        self.key_bits,
                    );
                }
                if self.target.quantize_values() {
                    self.decompress_array_bf16(
                        values,
                        &self.value_codes,
                        &self.value_norms,
                        self.value_row_bytes,
                        &self.value_codebook,
                        self.value_bits,
                    );
                }
            },
            DataType::F16 => {
                if self.target.quantize_keys() {
                    self.decompress_array_f16(
                        keys,
                        &self.key_codes,
                        &self.key_norms,
                        self.key_row_bytes,
                        &self.key_codebook,
                        self.key_bits,
                    );
                }
                if self.target.quantize_values() {
                    self.decompress_array_f16(
                        values,
                        &self.value_codes,
                        &self.value_norms,
                        self.value_row_bytes,
                        &self.value_codebook,
                        self.value_bits,
                    );
                }
            },
            DataType::F32 => {
                if self.target.quantize_keys() {
                    self.decompress_array_f32(
                        keys,
                        &self.key_codes,
                        &self.key_norms,
                        self.key_row_bytes,
                        &self.key_codebook,
                        self.key_bits,
                    );
                }
                if self.target.quantize_values() {
                    self.decompress_array_f32(
                        values,
                        &self.value_codes,
                        &self.value_norms,
                        self.value_row_bytes,
                        &self.value_codebook,
                        self.value_bits,
                    );
                }
            },
            dtype => panic!("TurboQuant does not support KV dtype {dtype:?}"),
        }
    }

    fn decompress_keys(
        &self,
        keys: &mut Array<B>,
    ) {
        assert_eq!(keys.shape(), &self.shape, "TurboQuant key shape mismatch");
        match keys.data_type() {
            DataType::BF16 => {
                if self.target.quantize_keys() {
                    self.decompress_array_bf16(
                        keys,
                        &self.key_codes,
                        &self.key_norms,
                        self.key_row_bytes,
                        &self.key_codebook,
                        self.key_bits,
                    );
                }
            },
            DataType::F16 => {
                if self.target.quantize_keys() {
                    self.decompress_array_f16(
                        keys,
                        &self.key_codes,
                        &self.key_norms,
                        self.key_row_bytes,
                        &self.key_codebook,
                        self.key_bits,
                    );
                }
            },
            DataType::F32 => {
                if self.target.quantize_keys() {
                    self.decompress_array_f32(
                        keys,
                        &self.key_codes,
                        &self.key_norms,
                        self.key_row_bytes,
                        &self.key_codebook,
                        self.key_bits,
                    );
                }
            },
            dtype => panic!("TurboQuant does not support KV dtype {dtype:?}"),
        }
    }

    fn decompress_values(
        &self,
        values: &mut Array<B>,
    ) {
        assert_eq!(values.shape(), &self.shape, "TurboQuant value shape mismatch");
        match values.data_type() {
            DataType::BF16 => {
                if self.target.quantize_values() {
                    self.decompress_array_bf16(
                        values,
                        &self.value_codes,
                        &self.value_norms,
                        self.value_row_bytes,
                        &self.value_codebook,
                        self.value_bits,
                    );
                }
            },
            DataType::F16 => {
                if self.target.quantize_values() {
                    self.decompress_array_f16(
                        values,
                        &self.value_codes,
                        &self.value_norms,
                        self.value_row_bytes,
                        &self.value_codebook,
                        self.value_bits,
                    );
                }
            },
            DataType::F32 => {
                if self.target.quantize_values() {
                    self.decompress_array_f32(
                        values,
                        &self.value_codes,
                        &self.value_norms,
                        self.value_row_bytes,
                        &self.value_codebook,
                        self.value_bits,
                    );
                }
            },
            dtype => panic!("TurboQuant does not support KV dtype {dtype:?}"),
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
        assert_eq!(keys.shape(), &self.shape, "TurboQuant key shape mismatch");
        assert_eq!(values.shape(), &self.shape, "TurboQuant value shape mismatch");
        assert_eq!(keys.data_type(), values.data_type(), "TurboQuant KV dtype mismatch");
        self.ensure_dense_storage(keys.data_type());

        match keys.data_type() {
            DataType::BF16 => {
                self.update_rows_from_dense_typed::<half::bf16>(keys, values, source_indices, destination_indices);
            },
            DataType::F16 => {
                self.update_rows_from_dense_typed::<half::f16>(keys, values, source_indices, destination_indices);
            },
            DataType::F32 => {
                self.update_rows_from_dense_typed::<f32>(keys, values, source_indices, destination_indices);
            },
            dtype => panic!("TurboQuant does not support KV dtype {dtype:?}"),
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
        assert_eq!(queries.len(), num_heads * head_dim, "TurboQuant query shape mismatch");
        assert_eq!(scores_out.len(), num_heads * prefix_length, "TurboQuant score shape mismatch");
        assert_eq!(num_heads % num_groups, 0, "TurboQuant GQA factor must divide head count");
        assert!(prefix_length <= sequence_length, "TurboQuant prefix length out of bounds");

        let gqa_factor = num_heads / num_groups;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let correction_scale = (PI / 2.0).sqrt() / head_dim as f32;
        let mut rotated_query = vec![0.0f32; head_dim];
        let mut projected_query = vec![0.0f32; head_dim];

        for head_index in 0..num_heads {
            let query = &queries[head_index * head_dim..(head_index + 1) * head_dim];
            rotate(&self.rotation, query, &mut rotated_query);
            rotate(&self.qjl_projection, query, &mut projected_query);

            let group_index = head_index / gqa_factor;
            for token_index in 0..prefix_length {
                let row_index = group_index * sequence_length + token_index;
                let scalar_score = self.key_norms[row_index]
                    * dot_packed_row(
                        &self.key_codes[row_index * self.key_row_bytes..(row_index + 1) * self.key_row_bytes],
                        &self.key_codebook,
                        self.key_bits,
                        &rotated_query,
                    );
                let residual_score = self.key_residual_norms[row_index]
                    * correction_scale
                    * dot_sign_row(
                        &self.key_residual_signs
                            [row_index * self.key_residual_row_bytes..(row_index + 1) * self.key_residual_row_bytes],
                        &projected_query,
                    );
                scores_out[head_index * prefix_length + token_index] = scale * (scalar_score + residual_score);
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
        assert!(self.target.quantize_values(), "TurboQuant value row decoding requires quantized values");
        assert!(row_index < self.shape[0] * self.shape[1], "TurboQuant value row index out of bounds");
        assert_eq!(output.len(), self.shape[2], "TurboQuant value row shape mismatch");
        assert_eq!(scratch.len(), output.len(), "TurboQuant value row scratch shape mismatch");

        decode_row(
            &self.value_codes[row_index * self.value_row_bytes..(row_index + 1) * self.value_row_bytes],
            &self.value_codebook,
            self.value_bits,
            scratch,
        );
        inverse_rotate(&self.rotation, scratch, output);
        for value in output.iter_mut() {
            *value *= self.value_norms[row_index];
        }
    }

    fn decode_value_rows_for_single_decode(
        &self,
        row_index: usize,
        row_count: usize,
        output: &mut [f32],
    ) {
        if row_count == 0 {
            assert!(output.is_empty(), "TurboQuant row-range output must be empty when row_count is zero");
            return;
        }
        assert!(self.target.quantize_values(), "TurboQuant value row decoding requires quantized values");
        assert!(row_index + row_count <= self.shape[0] * self.shape[1], "TurboQuant row range out of bounds");
        let head_dim = self.shape[2];
        assert_eq!(output.len(), row_count * head_dim, "TurboQuant row-range output shape mismatch");

        let mut scratch = vec![0.0f32; head_dim];
        for (offset, row) in output.chunks_exact_mut(head_dim).enumerate() {
            let row_index = row_index + offset;
            decode_row(
                &self.value_codes[row_index * self.value_row_bytes..(row_index + 1) * self.value_row_bytes],
                &self.value_codebook,
                self.value_bits,
                &mut scratch,
            );
            inverse_rotate(&self.rotation, &scratch, row);
            let norm = self.value_norms[row_index];
            for value in row.iter_mut() {
                *value *= norm;
            }
        }
    }

    fn memory_usage_bytes(&self) -> usize {
        let key_storage = if self.target.quantize_keys() {
            self.key_codes.len()
                + self.key_norms.len() * std::mem::size_of::<f32>()
                + self.key_residual_signs.len()
                + self.key_residual_norms.len() * std::mem::size_of::<f32>()
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

    let mut centroids: Vec<f32> = (0..levels)
        .map(|index| {
            let t = (index as f32 + 0.5) / levels as f32;
            (2.0 * t - 1.0) * 3.0
        })
        .collect();

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
            for (&x, &w) in grid.iter().zip(weights.iter()) {
                if x >= bounds[level] && x < bounds[level + 1] {
                    weighted_sum += x * w;
                    total_weight += w;
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

fn random_orthogonal(
    dim: usize,
    seed: u64,
) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut basis = vec![0.0f32; dim * dim];
    let mut vector = vec![0.0f32; dim];
    for column in 0..dim {
        loop {
            for value in &mut vector {
                *value = gaussian_sample(&mut rng);
            }
            for prev in 0..column {
                let mut dot = 0.0f32;
                for row in 0..dim {
                    dot += vector[row] * basis[prev * dim + row];
                }
                for row in 0..dim {
                    vector[row] -= dot * basis[prev * dim + row];
                }
            }
            let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
            if norm > 1e-5 {
                for row in 0..dim {
                    basis[column * dim + row] = vector[row] / norm;
                }
                break;
            }
        }
    }
    basis
}

fn scaled_random_orthogonal(
    dim: usize,
    seed: u64,
) -> Vec<f32> {
    let scale = (dim as f32).sqrt();
    random_orthogonal(dim, seed).into_iter().map(|value| value * scale).collect()
}

fn gaussian_sample(rng: &mut StdRng) -> f32 {
    let u1 = rng.random::<f32>().clamp(1e-7, 1.0 - 1e-7);
    let u2 = rng.random::<f32>();
    (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * PI * u2).cos()
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

fn encode_row(
    input: &[f32],
    codebook: &[f32],
    bits: usize,
    packed: &mut [u8],
) {
    packed.fill(0);
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

fn fill_sign_bits(
    projection: &[f32],
    input: &[f32],
    packed: &mut [u8],
) {
    packed.fill(0);
    let dim = input.len();
    let mut scratch = vec![0.0f32; dim];
    rotate(projection, input, &mut scratch);
    for (index, &value) in scratch.iter().enumerate() {
        if value >= 0.0 {
            write_bits(packed, index, 1, 1);
        }
    }
}

fn decode_row(
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
    let byte_index = bit_offset / 8;
    let shift = bit_offset % 8;
    let low = packed[byte_index] as u32;
    let high = packed.get(byte_index + 1).copied().unwrap_or(0) as u32;
    let mask = (1u32 << bits) - 1;
    ((low | (high << 8)) >> shift) & mask
}

fn dot_packed_row(
    packed: &[u8],
    codebook: &[f32],
    bits: usize,
    vector: &[f32],
) -> f32 {
    let mut sum = 0.0f32;
    let mut bit_offset = 0usize;
    for &value in vector {
        let index = read_bits(packed, bit_offset, bits) as usize;
        sum += codebook[index] * value;
        bit_offset += bits;
    }
    sum
}

fn dot_sign_row(
    packed: &[u8],
    vector: &[f32],
) -> f32 {
    vector
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if ((packed[index / 8] >> (index % 8)) & 1) == 1 {
                value
            } else {
                -value
            }
        })
        .sum()
}
