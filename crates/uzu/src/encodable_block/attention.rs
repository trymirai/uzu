//! Attention kernel encodable.

use std::collections::HashMap;

use itertools::iproduct;

use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::ring::RingParams,
        kernel::{
            AttentionSinglePassKernel, AttentionTwoPass1Kernel, AttentionTwoPass2Kernel, AttentionUpdateKVCacheKernel,
            BufferArg, BufferArgMut, SigmoidGateKernel, attention::AttentionGemmBlock,
        },
    },
};

fn env_gemm_attention_enabled() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        let raw = std::env::var("UZU_USE_GEMM_ATTENTION").ok();
        let Some(raw) = raw else {
            return true;
        };
        let v = raw.trim().to_ascii_uppercase();
        match v.as_str() {
            "1" | "YES" | "TRUE" | "ON" => true,
            "0" | "NO" | "FALSE" | "OFF" => false,
            _ => true,
        }
    })
}

pub struct Attention<B: Backend> {
    single_pass_kernels: HashMap<KernelKey, <B::Kernels as Kernels>::AttentionSinglePassKernel>,
    two_pass_1_kernels: HashMap<KernelKey, <B::Kernels as Kernels>::AttentionTwoPass1Kernel>,
    two_pass_2_kernels: HashMap<u32, <B::Kernels as Kernels>::AttentionTwoPass2Kernel>,
    update_kv_cache_kernel: <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel,
    update_kv_cache_inplace_kernel: <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel,
    gate_kernel: Option<<B::Kernels as Kernels>::SigmoidGateKernel>,
    gemm_block: AttentionGemmBlock<B>,
    layer_index: usize,
    attention_scale: Option<f32>,
    has_sinks: bool,
    is_causal: bool,
    sliding_window_size: Option<usize>,
}

pub struct AttentionArguments<A> {
    pub qkv: A,
    pub queries: A,
    pub rotated_keys: A,
    pub gate: Option<A>,
    pub trie: Option<A>,
    pub sinks: Option<A>,
    pub key_cache: Option<A>,
    pub value_cache: Option<A>,
    pub suffix_length: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_groups: usize,
    pub element_size: usize,
    pub segment_prefix_length: usize,
    pub max_sequence_length: usize,
    pub ring_params: Option<RingParams>,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        layer_index: usize,
        attention_scale: Option<f32>,
        has_sinks: bool,
        is_causal: bool,
        sliding_window_size: Option<usize>,
        has_gate: bool,
    ) -> Result<Self, B::Error> {
        let mut single_pass_kernels = HashMap::new();
        let mut two_pass_1_kernels = HashMap::new();
        let mut two_pass_2_kernels = HashMap::new();

        for (head_dim, is_trie, is_kv_cache_ring) in iproduct!([64u32, 128u32, 256u32], [false, true], [false, true]) {
            let key = KernelKey {
                head_dim,
                is_trie,
                is_kv_cache_ring,
            };

            let sp_kernel = <B::Kernels as Kernels>::AttentionSinglePassKernel::new(
                context,
                data_type,
                head_dim,
                has_sinks,
                is_kv_cache_ring,
                is_causal,
                is_trie,
                sliding_window_size.is_some(),
            )?;
            single_pass_kernels.insert(key, sp_kernel);

            let tp1_kernel = <B::Kernels as Kernels>::AttentionTwoPass1Kernel::new(
                context,
                data_type,
                head_dim,
                has_sinks,
                is_kv_cache_ring,
                is_causal,
                is_trie,
                sliding_window_size.is_some(),
            )?;
            two_pass_1_kernels.insert(key, tp1_kernel);

            let tp2_kernel = <B::Kernels as Kernels>::AttentionTwoPass2Kernel::new(context, data_type, head_dim)?;
            two_pass_2_kernels.insert(head_dim, tp2_kernel);
        }

        let update_kv_cache_kernel =
            <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel::new(context, data_type, false)?;
        let update_kv_cache_inplace_kernel =
            <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel::new(context, data_type, true)?;
        let gate_kernel = if has_gate {
            Some(<B::Kernels as Kernels>::SigmoidGateKernel::new(context, data_type)?)
        } else {
            None
        };
        let gemm_block = AttentionGemmBlock::new(data_type);

        Ok(Self {
            single_pass_kernels,
            two_pass_1_kernels,
            two_pass_2_kernels,
            update_kv_cache_kernel,
            update_kv_cache_inplace_kernel,
            gate_kernel,
            gemm_block,
            layer_index,
            attention_scale,
            has_sinks,
            is_causal,
            sliding_window_size,
        })
    }

    fn select_variant(
        &self,
        gemm_enabled: bool,
        suffix_length: usize,
        head_dim: usize,
        sequence_length: usize,
        is_trie: bool,
        is_kv_cache_ring: bool,
    ) -> KernelVariant {
        let use_gemm = gemm_enabled && suffix_length > 8 && matches!(head_dim, 64 | 128 | 256);
        if use_gemm {
            return KernelVariant::Gemm;
        }

        let kernel_key = KernelKey {
            head_dim: head_dim as u32,
            is_trie,
            is_kv_cache_ring,
        };
        if sequence_length > 1024
            && self.two_pass_1_kernels.contains_key(&kernel_key)
            && self.two_pass_2_kernels.contains_key(&(head_dim as u32))
        {
            return KernelVariant::TwoPass;
        }

        KernelVariant::SinglePass
    }

    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub fn has_gate(&self) -> bool {
        self.gate_kernel.is_some()
    }

    pub fn has_sinks(&self) -> bool {
        self.has_sinks
    }

    pub fn encode<A>(
        &self,
        context: &B::Context,
        runtime: &mut AttentionArguments<A>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error>
    where
        for<'a> &'a A: BufferArg<'a, B::Buffer>,
        for<'a> &'a mut A: BufferArgMut<'a, B::Buffer>,
    {
        let qkv = &runtime.qkv;
        let queries = &runtime.queries;
        let rotated_keys = &mut runtime.rotated_keys;
        let gate = runtime.gate.as_ref();
        let trie = runtime.trie.as_ref();
        let sinks = runtime.sinks.as_ref();
        let key_cache = &mut runtime.key_cache;
        let value_cache = &mut runtime.value_cache;
        let suffix_length = runtime.suffix_length;
        let num_heads = runtime.num_heads;
        let head_dim = runtime.head_dim;
        let num_groups = runtime.num_groups;
        let element_size = runtime.element_size;
        let segment_prefix_length = runtime.segment_prefix_length;
        let max_sequence_length = runtime.max_sequence_length;
        let ring_params = runtime.ring_params;

        let is_trie = trie.is_some();
        let has_kv_cache = key_cache.is_some();
        let is_kv_cache_ring = ring_params.is_some();
        let sequence_length = segment_prefix_length + suffix_length;

        let gqa_factor = num_heads / num_groups;
        let scale = self.attention_scale.unwrap_or(1.0f32 / (head_dim as f32).sqrt());

        let gemm_enabled = env_gemm_attention_enabled();
        if !gemm_enabled {
            static PRINT_ONCE: std::sync::Once = std::sync::Once::new();
            PRINT_ONCE.call_once(|| {
                eprintln!("[uzu] Gemm attention disabled via UZU_USE_GEMM_ATTENTION");
            });
        }
        let variant =
            self.select_variant(gemm_enabled, suffix_length, head_dim, sequence_length, is_trie, is_kv_cache_ring);

        let mut attention_output = encoder.allocate_scratch(suffix_length * num_heads * head_dim * element_size)?;
        let mut extracted_values = if has_kv_cache {
            None
        } else {
            Some(encoder.allocate_scratch(suffix_length * num_groups * head_dim * element_size)?)
        };

        // For classifiers (no KV cache): extract values from QKV into a dedicated extracted_values buffer.
        if let Some(extracted_values) = extracted_values.as_mut() {
            self.update_kv_cache_inplace_kernel.encode(
                None::<&Allocation<B>>,
                qkv,
                &mut *rotated_keys,
                extracted_values,
                num_groups as u32,
                num_heads as u32,
                head_dim as u32,
                suffix_length as u32,
                0u32,
                max_sequence_length as u32,
                encoder,
            );
        }

        if has_kv_cache {
            self.update_kv_cache_kernel.encode(
                Some(&*rotated_keys),
                qkv,
                key_cache.as_mut().unwrap(),
                value_cache.as_mut().unwrap(),
                num_groups as u32,
                num_heads as u32,
                head_dim as u32,
                suffix_length as u32,
                segment_prefix_length as u32,
                max_sequence_length as u32,
                encoder,
            );
        }

        let key_cache_allocation = if has_kv_cache {
            key_cache.as_ref().unwrap()
        } else {
            &*rotated_keys
        };
        let sliding_window_size = self.sliding_window_size.map(|s| s as u32);

        let k_head_stride = (max_sequence_length * head_dim) as i32;
        let k_seq_stride = head_dim as i32;
        let v_head_stride = (max_sequence_length * head_dim) as i32;
        let v_seq_stride = head_dim as i32;

        let kernel_key = KernelKey {
            head_dim: head_dim as u32,
            is_trie,
            is_kv_cache_ring,
        };

        macro_rules! with_values {
            ($values:ident => $expr:expr) => {
                match value_cache.as_ref() {
                    Some($values) => $expr,
                    None => {
                        let $values = extracted_values.as_ref().unwrap();
                        $expr
                    },
                }
            };
        }

        match variant {
            KernelVariant::Gemm => with_values!(values => {
                self.gemm_block
                    .encode(
                        context,
                        encoder,
                        queries,
                        key_cache_allocation,
                        values,
                        &mut attention_output,
                        trie,
                        sinks,
                        num_heads,
                        num_groups,
                        suffix_length,
                        sequence_length,
                        segment_prefix_length,
                        max_sequence_length,
                        ring_params,
                        head_dim,
                        self.sliding_window_size,
                        self.is_causal,
                        scale,
                    )
                    .expect("Failed to encode AttentionGemmBlock")
            }),
            KernelVariant::SinglePass => {
                let kernel = match self.single_pass_kernels.get(&kernel_key) {
                    Some(k) => k,
                    None => panic!("Can not find AttentionSinglePassKernel for key {:?}", kernel_key),
                };
                with_values!(values => {
                    kernel.encode(
                        queries,
                        key_cache_allocation,
                        values,
                        &mut attention_output,
                        gqa_factor as u32,
                        sequence_length as u32,
                        k_head_stride as u32,
                        k_seq_stride as u32,
                        v_head_stride as u32,
                        v_seq_stride as u32,
                        ring_params,
                        scale,
                        trie,
                        sliding_window_size,
                        sinks,
                        num_heads as u32,
                        suffix_length as u32,
                        encoder,
                    )
                })
            },
            KernelVariant::TwoPass => {
                let kernel_pass1 = match self.two_pass_1_kernels.get(&kernel_key) {
                    Some(k) => k,
                    None => panic!("Can not find AttentionTwoPass1Kernel for key {:?}", kernel_key),
                };
                let kernel_pass2 = match self.two_pass_2_kernels.get(&(head_dim as u32)) {
                    Some(k) => k,
                    None => panic!("Can not find AttentionTwoPass2Kernel for key {:?}", kernel_key),
                };
                let mut partials =
                    encoder.allocate_scratch(suffix_length * num_heads * 32 * head_dim * element_size)?;
                let mut sums = encoder.allocate_scratch(suffix_length * num_heads * 32 * element_size)?;
                let mut maxs = encoder.allocate_scratch(suffix_length * num_heads * 32 * element_size)?;
                with_values!(values => {
                    kernel_pass1.encode(
                        queries,
                        key_cache_allocation,
                        values,
                        &mut partials,
                        &mut sums,
                        &mut maxs,
                        gqa_factor as u32,
                        sequence_length as u32,
                        k_head_stride as u32,
                        k_seq_stride as u32,
                        v_head_stride as u32,
                        v_seq_stride as u32,
                        ring_params,
                        scale,
                        num_heads as u32,
                        suffix_length as u32,
                        trie,
                        sliding_window_size,
                        sinks,
                        encoder,
                    );
                });
                kernel_pass2.encode(
                    &partials,
                    &sums,
                    &maxs,
                    &mut attention_output,
                    num_heads as u32,
                    suffix_length as u32,
                    encoder,
                );
            },
        }

        if let Some(gate_kernel) = &self.gate_kernel {
            let gate = gate.expect("Gate buffer missing for gated attention");
            let total_elements = (suffix_length * num_heads * head_dim) as u32;
            gate_kernel.encode(gate, &mut attention_output, total_elements, encoder);
        }

        Ok(attention_output)
    }
}

enum KernelVariant {
    Gemm,
    SinglePass,
    TwoPass,
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
struct KernelKey {
    pub head_dim: u32,
    pub is_trie: bool,
    pub is_kv_cache_ring: bool,
}

#[cfg(test)]
#[path = "../../tests/unit/encodable_block/attention_test.rs"]
mod tests;
