use half::bf16;
use num_traits::AsPrimitive;

use crate::{
    backends::{
        common::{
            BufferRef, CommandBufferEncoding,
            gpu_types::{kv_layout, trie::TrieNode},
            kernel::{AttentionArguments, AttentionInput, AttentionKernelConfig},
        },
        cpu::{
            Cpu,
            buffer::CpuBufferExt,
            command_buffer::CpuCommandBufferEncoding,
            error::CpuError,
            kernel::{activation_transform::hadamard_transform, attention::mask::should_use_key},
        },
    },
    utils::{
        dispatch_dtype::dispatch_dtype,
        pointers::{SendPtr, SendPtrMut},
    },
};

pub(super) fn sign(channel: usize) -> f32 {
    if (kv_layout::ROTATION_SIGN_BITS[channel / 32 % 8] >> (channel % 32)) & 1 != 0 {
        -1.0
    } else {
        1.0
    }
}

pub(super) fn quantize(mut values: Vec<f32>) -> (Vec<i8>, bf16) {
    for (channel, value) in values.iter_mut().enumerate() {
        *value *= sign(channel);
    }
    hadamard_transform(&mut values);
    let maximum = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = bf16::from_f32(if maximum > 0.0 {
        maximum / 127.0
    } else {
        1.0
    });
    (values.into_iter().map(|v| (v / scale.to_f32()).round_ties_even().clamp(-127.0, 127.0) as i8).collect(), scale)
}

pub(super) struct Row {
    maximum: f32,
    denominator: f32,
    values: Vec<f32>,
}

impl Row {
    pub(super) fn new(
        head_dim: usize,
        sink: Option<f32>,
    ) -> Self {
        Self {
            maximum: sink.unwrap_or(f32::NEG_INFINITY),
            denominator: if sink.is_some() {
                1.0
            } else {
                0.0
            },
            values: vec![0.0; head_dim],
        }
    }

    pub(super) fn add(
        &mut self,
        score: f32,
        value: impl Fn(usize) -> f32,
    ) {
        let maximum = self.maximum.max(score);
        let correction = (self.maximum - maximum).exp();
        let weight = (score - maximum).exp();
        self.denominator = self.denominator * correction + weight;
        for (channel, sum) in self.values.iter_mut().enumerate() {
            *sum = *sum * correction + weight * value(channel);
        }
        self.maximum = maximum;
    }

    pub(super) fn finish(mut self) -> Vec<f32> {
        if self.denominator > 0.0 {
            for value in &mut self.values {
                *value /= self.denominator;
            }
        }
        hadamard_transform(&mut self.values);
        for (channel, value) in self.values.iter_mut().enumerate() {
            *value *= sign(channel);
        }
        self.values
    }
}

pub(super) fn encode(
    input: AttentionInput<impl BufferRef<Backend = Cpu>, impl BufferRef<Backend = Cpu>, impl BufferRef<Backend = Cpu>>,
    arguments: AttentionArguments<'_, Cpu, impl BufferRef<Backend = Cpu>, impl BufferRef<Backend = Cpu>>,
    config: AttentionKernelConfig,
    command_buffer: &mut CpuCommandBufferEncoding,
) -> Result<<Cpu as crate::backends::common::Backend>::ScratchBuffer, CpuError> {
    fn ptr(view: impl BufferRef<Backend = Cpu>) -> SendPtr<u8> {
        let (buffer, range) = view.parts();
        SendPtr(buffer.cpu_address().as_ptr().cast::<u8>().cast_const().wrapping_add(range.start))
    }
    let output = command_buffer.allocate_scratch_for_shape(
        &[arguments.suffix_length, config.num_q_heads, config.head_dim],
        config.data_type,
    )?;
    let output_ptr = SendPtrMut(ptr(&output).as_ptr().cast_mut());
    let q_codes = ptr(input.query_codes);
    let q_scales = ptr(input.query_scales);
    let k_codes = ptr(input.k_codes);
    let k_scales = ptr(input.k_scales);
    let v_codes = ptr(input.v_codes);
    let v_scales = ptr(input.v_scales);
    let trie = arguments.trie.map(ptr);
    let sinks = arguments.sinks.map(ptr);
    let gate = arguments.gate.map(ptr);
    let prefix = arguments.cache.prefix_len();
    let ring = arguments.cache.ring_params();
    let suffix = arguments.suffix_length;
    command_buffer.push_command(move || {
        let dim = config.head_dim as usize;
        let gqa = config.num_q_heads / config.num_groups;
        let scale = config.scale.unwrap_or(1.0 / (dim as f32).sqrt());
        let suffix_position = ring.map_or(prefix, |r| r.ring_length);
        let trie = trie.map(|p| p.as_ptr().cast::<TrieNode>());
        dispatch_dtype!(|(T: config.data_type)| {
            for head in 0..config.num_q_heads {
                let kv_head = head / gqa;
                for query in 0..suffix {
                    let qrow = (head * suffix + query) as usize;
                    unsafe {
                        let qs = (*q_scales.as_ptr().cast::<bf16>().add(qrow)).to_f32();
                        let position = suffix_position + trie.map_or(query, |p| (*p.add(query as usize)).height);
                        let mut row = Row::new(dim, sinks.map(|p| (*p.as_ptr().cast::<T>().add(head as usize)).as_()));
                        for key in 0..prefix + suffix {
                            if !should_use_key(
                                ring, trie, config.sliding_window_size, query, prefix,
                                suffix_position, position, key, config.is_causal,
                            ) {
                                continue;
                            }
                            let kv_row = (key * config.num_groups + kv_head) as usize;
                            let dot: i32 = (0..dim)
                                .map(|c| {
                                    let q = *q_codes.as_ptr().cast::<i8>().add(qrow * dim + c);
                                    let k = *k_codes.as_ptr().cast::<i8>()
                                        .add(kv_row * dim + c);
                                    i32::from(q) * i32::from(k)
                                })
                                .sum();
                            let ks = (*k_scales.as_ptr().cast::<bf16>().add(kv_row)).to_f32();
                            let vs = (*v_scales.as_ptr().cast::<bf16>().add(kv_row)).to_f32();
                            row.add(dot as f32 * qs * ks * scale, |c| {
                                let code = *v_codes.as_ptr().add(kv_row * dim + c);
                                (i32::from(code) - 128) as f32 * vs
                            });
                        }
                        for (c, mut value) in row.finish().into_iter().enumerate() {
                            if let (Some(gate), Some(stride)) = (gate, config.gate_row_stride) {
                                let index = (query * stride + head * config.head_dim) as usize + c;
                                let g: f32 = (*gate.as_ptr().cast::<T>().add(index)).as_();
                                let rounded: T = value.as_();
                                value = rounded.as_();
                                value /= 1.0 + (-g).exp();
                            }
                            let index = (query * config.num_q_heads + head) as usize * dim + c;
                            *output_ptr.as_ptr().cast::<T>().add(index) = value.as_();
                        }
                    }
                }
            }
        });
    });
    Ok(output)
}
