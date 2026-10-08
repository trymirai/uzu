use std::{iter::repeat_n, mem::size_of};

use bytemuck::NoUninit;
use half::f16;
use num_traits::Float;

use super::{kernel_fixture::KernelFixture, round32, signs, to, transform_oracle, values};
use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context, Kernels,
            gpu_types::{EmbeddingTableKind, QuantizationMethod, QuantizationMode, d4s4},
            kernel::InputEmbeddingLookupKernel,
        },
        cpu::Cpu,
        vulkan::{InputEmbeddingLookupVulkanKernel, VkCommandBufferEncoding},
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// Guard bytes around every read-only payload.
const INPUT_SENTINEL: u8 = 0xa5;
/// Output elements past the batch, which every dispatch must leave untouched.
const TAIL: usize = 5;

/// One raw input embedding lookup shared by the CPU and Vulkan runners and the staged oracle: the specialization, the
/// dimensions and every payload in its canonical layout, optional ones present exactly as the signature requires.
#[derive(Clone)]
pub struct InputEmbeddingLookupCase<T> {
    pub table_kind: EmbeddingTableKind,
    pub quantization: Option<(QuantizationMode, QuantizationMethod, u32)>,
    pub token_ids: Vec<u32>,
    pub vocab_size: u32,
    pub model_dim: u32,
    pub input_scale: f32,
    /// Dense T[vocab, D] bytes, quantized codes [vocab, D / packing divisor] or D4S4 codes [vocab, D / 4].
    pub values: Vec<u8>,
    /// Quantized T[vocab, groups] or D4S4 T[vocab].
    pub scales: Option<Vec<T>>,
    /// [vocab, groups] bytes, U4 packed per row into [vocab, ceil(groups / 2)].
    pub zero_points: Option<Vec<u8>>,
    pub biases: Option<Vec<T>>,
    pub factors: Option<Vec<i32>>,
    /// D4S4 nibbles packed over the whole table: token * D / 64 + column / 64.
    pub ladder_indices: Option<Vec<u8>>,
    pub ladder: Option<Vec<f16>>,
    pub codebook: Option<Vec<i8>>,
}

impl<T: ArrayElement + Float + NoUninit> InputEmbeddingLookupCase<T> {
    /// Deterministic payloads of every byte value and mixed-sign T values for `token_ids` rows of `model_dim`.
    pub fn new(
        table_kind: EmbeddingTableKind,
        quantization: Option<(QuantizationMode, QuantizationMethod, u32)>,
        (vocab_size, model_dim): (u32, u32),
        token_ids: &[u32],
        seed: usize,
    ) -> Self {
        let (vocab, dim) = (vocab_size as usize, model_dim as usize);
        let bytes = |length: usize, salt: usize| {
            (0..length).map(|i| ((i * 167 + (seed + salt) * 59) % 256) as u8).collect::<Vec<_>>()
        };
        let mut case = Self {
            table_kind,
            quantization,
            token_ids: token_ids.to_vec(),
            vocab_size,
            model_dim,
            input_scale: 1.37,
            values: Vec::new(),
            scales: None,
            zero_points: None,
            biases: None,
            factors: None,
            ladder_indices: None,
            ladder: None,
            codebook: None,
        };
        match (table_kind, quantization) {
            (EmbeddingTableKind::Dense, None) => {
                case.values = bytemuck::cast_slice(&values::<T>(vocab * dim, seed)).to_vec();
            },
            (EmbeddingTableKind::Quantized, Some((mode, method, group_size))) => {
                let groups = vocab * dim.div_ceil(group_size as usize);
                case.values = bytes(vocab * dim / mode.packing_divisor() as usize, 0);
                case.scales = Some(values(groups, seed + 1));
                case.biases = (method == QuantizationMethod::ScaleBias).then(|| values(groups, seed + 2));
                case.zero_points = (method == QuantizationMethod::ScaleZeroPoint).then(|| match mode {
                    QuantizationMode::U4 => bytes(vocab * dim.div_ceil(group_size as usize).div_ceil(2), 3),
                    _ => bytes(groups, 3),
                });
            },
            (EmbeddingTableKind::D4S4, None) => {
                let ladder_groups = vocab * dim / d4s4::COLUMNS_PER_LADDER_SCALE as usize;
                case.values = bytes(vocab * dim / d4s4::VALUES_PER_CODE as usize, 0);
                case.scales = Some(values(vocab, seed + 1));
                case.ladder_indices = Some(bytes(ladder_groups.div_ceil(2), 4));
                case.ladder = Some((0..d4s4::LADDER_SIZE).map(|i| f16::from_f32(1.5f32.powi(i as i32 - 9))).collect());
                case.codebook =
                    Some((0..d4s4::CODEBOOK_SIZE * d4s4::VALUES_PER_CODE).map(|i| (i * 37) as u8 as i8).collect());
            },
            _ => panic!("{table_kind:?} with {quantization:?}"),
        }
        case
    }

    /// With output Hadamard factors, deterministic signs.
    pub fn hadamard(
        mut self,
        seed: usize,
    ) -> Self {
        self.factors = Some(signs(self.model_dim as usize, seed));
        self
    }

    pub fn label(&self) -> String {
        format!(
            "{:?} {:?} {:?} {}x{} vocab {} scale {} hadamard {}",
            T::data_type(),
            self.table_kind,
            self.quantization,
            self.token_ids.len(),
            self.model_dim,
            self.vocab_size,
            self.input_scale,
            self.factors.is_some()
        )
    }

    pub fn settings(&self) -> (Option<u32>, Option<QuantizationMode>, Option<QuantizationMethod>) {
        self.quantization
            .map_or((None, None, None), |(mode, method, group_size)| (Some(group_size), Some(mode), Some(method)))
    }

    pub fn vulkan_kernel(
        &self,
        fixture: &KernelFixture,
    ) -> InputEmbeddingLookupVulkanKernel {
        let (group_size, mode, method) = self.settings();
        let use_hadamard = self.factors.is_some();
        InputEmbeddingLookupVulkanKernel::new(
            &fixture.context,
            T::data_type(),
            self.table_kind,
            group_size,
            mode,
            method,
            use_hadamard,
        )
        .expect("Vulkan InputEmbeddingLookup")
    }

    /// The read-only payloads as bytes in signature order, absent optional ones `None`.
    pub fn inputs(&self) -> [Option<Vec<u8>>; 9] {
        fn bytes<U: NoUninit>(values: &[U]) -> Vec<u8> {
            bytemuck::cast_slice(values).to_vec()
        }
        [
            Some(bytes(&self.token_ids)),
            Some(self.values.clone()),
            self.scales.as_deref().map(bytes),
            self.zero_points.clone(),
            self.biases.as_deref().map(bytes),
            self.factors.as_deref().map(bytes),
            self.ladder_indices.clone(),
            self.ladder.as_deref().map(bytes),
            self.codebook.as_deref().map(bytes),
        ]
    }

    /// The staged lookup in FP64: each FP32 operation of the canonical order and each rounding to T applied exactly as
    /// rounding, then the output transform oracle and the rounding to T. Tokens outside the vocabulary give +0.
    pub fn oracle(&self) -> Vec<((f64, f64), f64)> {
        let dim = self.model_dim as usize;
        let input_scale = f64::from(self.input_scale);
        let wide = |value: &T| value.to_f64().unwrap();
        let nibble = |bytes: &[u8], index: usize| f64::from(bytes[index / 2] >> (4 * (index % 2)) & 15);
        let dense = bytemuck::pod_collect_to_vec::<u8, T>(&self.values);
        let staged = |token: usize, column: usize| match self.table_kind {
            EmbeddingTableKind::Dense => to::<T>(round32(wide(&dense[token * dim + column]) * to::<T>(input_scale))),
            EmbeddingTableKind::Quantized => {
                let (mode, method, group_size) = self.quantization.expect("quantization settings");
                let groups = dim.div_ceil(group_size as usize);
                let group = column / group_size as usize;
                let scale = wide(&self.scales.as_ref().unwrap()[token * groups + group]);
                let row = token * (dim / mode.packing_divisor() as usize);
                let code = match mode {
                    QuantizationMode::U4 => nibble(&self.values, 2 * row + column),
                    QuantizationMode::I8 => f64::from(self.values[row + column] as i8),
                    QuantizationMode::U8 => f64::from(self.values[row + column]),
                };
                let bias = match (method, mode) {
                    (QuantizationMethod::ScaleBias, _) => wide(&self.biases.as_ref().unwrap()[token * groups + group]),
                    (QuantizationMethod::ScaleZeroPoint, QuantizationMode::U4) => -round32(
                        scale * nibble(self.zero_points.as_ref().unwrap(), 2 * token * groups.div_ceil(2) + group),
                    ),
                    (QuantizationMethod::ScaleZeroPoint, _) => {
                        -round32(scale * f64::from(self.zero_points.as_ref().unwrap()[token * groups + group]))
                    },
                    (QuantizationMethod::ScaleSymmetric, QuantizationMode::U4) => -round32(scale * 8.0),
                    (QuantizationMethod::ScaleSymmetric, _) => -round32(scale * 128.0),
                };
                to::<T>(round32(round32(round32(scale * code) + bias) * input_scale))
            },
            EmbeddingTableKind::D4S4 => {
                let (per_code, per_scale) = (d4s4::VALUES_PER_CODE as usize, d4s4::COLUMNS_PER_LADDER_SCALE as usize);
                let ladder_index =
                    nibble(self.ladder_indices.as_ref().unwrap(), token * (dim / per_scale) + column / per_scale);
                let code = self.values[token * (dim / per_code) + column / per_code] as usize;
                let point = f64::from(self.codebook.as_ref().unwrap()[per_code * code + column % per_code]);
                let step = self.ladder.as_ref().unwrap()[ladder_index as usize].to_f64();
                let row_scale = wide(&self.scales.as_ref().unwrap()[token]);
                round32(round32(round32(row_scale * step) * point) * input_scale)
            },
        };
        let mut result = Vec::with_capacity(self.token_ids.len() * dim);
        for &token in &self.token_ids {
            if token >= self.vocab_size {
                result.extend(repeat_n(((0.0, 0.0), 0.0), dim));
                continue;
            }
            let row = (0..dim).map(|column| staged(token as usize, column)).map(|value| ((value, value), value));
            match &self.factors {
                Some(factors) => result.extend(transform_oracle(&row.collect::<Vec<_>>(), factors, false)),
                None => result.extend(row),
            }
        }
        result.into_iter().map(|((lo, hi), center)| ((to::<T>(lo), to::<T>(hi)), to::<T>(center))).collect()
    }

    /// The CPU kernel through the shared trait. CPU buffers cannot be empty, so every payload is followed by one byte
    /// the lookup never reads, which an empty vocabulary's tables then consist of.
    pub fn cpu(&self) -> Vec<T> {
        let count = self.token_ids.len() * self.model_dim as usize;
        if count == 0 {
            return Vec::new();
        }
        let context = create_context::<Cpu>();
        let (group_size, mode, method) = self.settings();
        let kernel = <<Cpu as Backend>::Kernels as Kernels>::InputEmbeddingLookupKernel::new(
            &context,
            T::data_type(),
            self.table_kind,
            group_size,
            mode,
            method,
            self.factors.is_some(),
        )
        .expect("CPU InputEmbeddingLookup");
        let [token_ids, values, scales, zero_points, biases, factors, ladder_indices, ladder, codebook] = self
            .inputs()
            .map(|input| input.map(|bytes| create_buffer_with_data::<Cpu, u8>(&context, &[bytes, vec![0]].concat())));
        let mut output = create_buffer_with_data::<Cpu, T>(&context, &vec![T::zero(); count]);
        let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
        kernel.encode(
            token_ids.as_ref().unwrap(),
            values.as_ref().unwrap(),
            scales.as_ref(),
            zero_points.as_ref(),
            biases.as_ref(),
            factors.as_ref(),
            ladder_indices.as_ref(),
            ladder.as_ref(),
            codebook.as_ref(),
            &mut output,
            self.token_ids.len() as u32,
            self.vocab_size,
            self.model_dim,
            self.input_scale,
            &mut command_buffer,
        );
        submit_command_buffer(command_buffer);
        buffer_to_vec::<Cpu, T>(&output)
    }

    /// Records every case into `encoding` over guarded ranges at nonzero offsets, the outputs followed by `TAIL`
    /// elements, and completes it. Returns each case's outputs after checking every guard, the untouched tails and the
    /// unchanged read-only payloads.
    pub fn gpu(
        fixture: &KernelFixture,
        cases: &[Self],
        mut encoding: VkCommandBufferEncoding,
    ) -> Vec<Vec<T>> {
        let sentinel = T::from(-7.0).unwrap();
        let prepared = cases
            .iter()
            .map(|case| {
                let count = case.token_ids.len() * case.model_dim as usize;
                let inputs = case.inputs().map(|input| input.map(|bytes| fixture.guarded(&bytes, INPUT_SENTINEL)));
                (case.vulkan_kernel(fixture), inputs, fixture.guarded(&vec![sentinel; count + TAIL], sentinel))
            })
            .collect::<Vec<_>>();
        for (case, (kernel, inputs, (output, range))) in cases.iter().zip(&prepared) {
            let [token_ids, values, scales, zero_points, biases, factors, ladder_indices, ladder, codebook] =
                inputs.each_ref().map(|input| input.as_ref().map(|(buffer, range)| (buffer, range.clone())));
            let end = range.end - (TAIL * size_of::<T>()) as u64;
            // SAFETY: each range holds exactly its canonical payload for these dimensions, the output range one T per
            // element of the batch, and no range aliases another.
            unsafe {
                kernel.encode(
                    token_ids.unwrap(),
                    values.unwrap(),
                    scales,
                    zero_points,
                    biases,
                    factors,
                    ladder_indices,
                    ladder,
                    codebook,
                    (output, range.start..end),
                    case.token_ids.len() as u32,
                    case.vocab_size,
                    case.model_dim,
                    case.input_scale,
                    &mut encoding,
                );
            }
        }
        KernelFixture::complete(encoding);
        cases
            .iter()
            .zip(&prepared)
            .map(|(case, (_, inputs, output))| {
                // SAFETY: the only command buffer using these buffers has completed.
                unsafe {
                    for ((input, payload), name) in inputs.iter().zip(case.inputs()).zip([
                        "token_ids",
                        "values",
                        "scales",
                        "zero_points",
                        "biases",
                        "factors",
                        "ladder_indices",
                        "ladder",
                        "codebook",
                    ]) {
                        if let (Some(input), Some(payload)) = (input, payload) {
                            KernelFixture::assert_unchanged(input, INPUT_SENTINEL, &payload, name);
                        }
                    }
                    let mut values = KernelFixture::read_guarded(output, sentinel);
                    let tail = values.split_off(values.len() - TAIL);
                    let unchanged = tail.iter().all(|value| bytemuck::bytes_of(value) == bytemuck::bytes_of(&sentinel));
                    assert!(unchanged, "{}: output tail written", case.label());
                    values
                }
            })
            .collect()
    }
}
