//! Attention kernel encodable.

use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer, Kernels,
        kernel::{
            AttentionSinglePassKernel, AttentionTwoPass1Kernel, AttentionTwoPass2Kernel, AttentionUpdateKVCacheKernel,
            attention::{AttentionGemmArguments, AttentionGemmBlock},
        },
    },
    encodable_block::EncodingParameters,
    forward_pass::state::{ArrayId, ForwardPassState, HashMapId},
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
    gemm_block: AttentionGemmBlock<B>,
    layer_index: usize,
    attention_scale: Option<f32>,
    has_sinks: bool,
    is_causal: bool,
    sliding_window_size: Option<usize>,
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
    ) -> Result<Self, B::Error> {
        let mut single_pass_kernels = HashMap::new();
        let mut two_pass_1_kernels = HashMap::new();
        let mut two_pass_2_kernels = HashMap::new();

        let supported_head_dims = [64u32, 128u32, 256u32];
        for head_dim in supported_head_dims {
            for has_mask in [false, true] {
                let key = KernelKey {
                    head_dim,
                    has_mask,
                };
                let float_mask = has_mask;

                let sp_kernel = <B::Kernels as Kernels>::AttentionSinglePassKernel::new(
                    context, data_type, head_dim, float_mask, has_mask, has_sinks, is_causal,
                )?;
                single_pass_kernels.insert(key, sp_kernel);

                let tp1_kernel = <B::Kernels as Kernels>::AttentionTwoPass1Kernel::new(
                    context, data_type, head_dim, float_mask, has_mask, has_sinks, is_causal,
                )?;
                two_pass_1_kernels.insert(key, tp1_kernel);

                let tp2_kernel = <B::Kernels as Kernels>::AttentionTwoPass2Kernel::new(context, data_type, head_dim)?;
                two_pass_2_kernels.insert(head_dim, tp2_kernel);
            }
        }

        let update_kv_cache_kernel =
            <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel::new(context, data_type, false)?;
        let update_kv_cache_inplace_kernel =
            <B::Kernels as Kernels>::AttentionUpdateKVCacheKernel::new(context, data_type, true)?;
        let gemm_block = AttentionGemmBlock::new(data_type);

        Ok(Self {
            single_pass_kernels,
            two_pass_1_kernels,
            two_pass_2_kernels,
            update_kv_cache_kernel,
            update_kv_cache_inplace_kernel,
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
        use_mask: bool,
    ) -> KernelVariant {
        let use_gemm = gemm_enabled && suffix_length > 8 && matches!(head_dim, 64 | 128 | 256);
        if use_gemm {
            return KernelVariant::Gemm;
        }

        let kernel_key = KernelKey {
            head_dim: head_dim as u32,
            has_mask: use_mask,
        };
        if sequence_length > 1024
            && self.two_pass_1_kernels.contains_key(&kernel_key)
            && self.two_pass_2_kernels.contains_key(&(head_dim as u32))
        {
            return KernelVariant::TwoPass;
        }

        KernelVariant::SinglePass
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        parameters: &EncodingParameters,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let (suffix_length, num_heads, head_dim, num_groups, max_sequence_length) = {
            let qkv_binding = state.arrays(&[ArrayId::QKV]);
            let qkv_array = qkv_binding[0].borrow();
            let suffix_length = qkv_array.shape()[0];

            let queries_binding = state.arrays(&[ArrayId::RotatedQueries]);
            let queries_array = queries_binding[0].borrow();
            let num_heads = queries_array.shape()[0];
            let head_dim = queries_array.shape()[2];

            let keys_binding = state.arrays(&[ArrayId::RotatedKeys]);
            let keys_array = keys_binding[0].borrow();
            let num_groups = keys_array.shape()[0];

            let max_sequence_length = if let Some(_kv) = state.cache_layers() {
                let key_cache_binding = state.arrays(&[ArrayId::Keys(self.layer_index)]);
                let key_cache_array = key_cache_binding[0].borrow();
                key_cache_array.shape()[1]
            } else {
                // For classifiers without KV cache, max_sequence_length is just suffix_length
                suffix_length
            };

            (suffix_length, num_heads, head_dim, num_groups, max_sequence_length)
        };

        let (segment_prefix_length, window_length) = if let Some(cache_layers) = state.cache_layers() {
            let cache = cache_layers.borrow();
            let layer = cache.data[self.layer_index]
                .as_transformer()
                .expect("Attention kernel expects transformer layer state");
            (layer.projected_segment_prefix_length(parameters.projection_step.unwrap_or(0)), layer.window_length())
        } else {
            // For classifiers without KV cache: no prefix, use configured sliding window
            (0, self.sliding_window_size)
        };

        let use_mask = window_length.is_some();

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
        let variant = self.select_variant(gemm_enabled, suffix_length, head_dim, sequence_length, use_mask);

        let rotated_queries_binding = state.arrays(&[ArrayId::RotatedQueries]);
        let rotated_keys_binding = state.arrays(&[ArrayId::RotatedKeys]);
        let qkv_binding = state.arrays(&[ArrayId::QKV]);
        let attention_bias_binding = state.hashmaps(&[HashMapId::AttentionBias]);
        let attention_output_binding = state.arrays(&[ArrayId::AttentionOutput]);

        let mask_kv_seq_stride = 1;
        let mask_q_seq_stride = sequence_length as i32;
        let mask_head_stride = 0;
        let sinks_binding = if self.has_sinks {
            Some(state.arrays(&[ArrayId::AttentionSinks(self.layer_index)]))
        } else {
            None
        };

        let rotated_keys_array = rotated_keys_binding[0].borrow_mut();
        let rotated_keys_buf_rc = rotated_keys_array.buffer();
        let mut rotated_keys_buf_borrow = rotated_keys_buf_rc.borrow_mut();

        let qkv_array = qkv_binding[0].borrow_mut();
        let qkv_buf_rc = qkv_array.buffer();
        let qkv_buf_borrow = qkv_buf_rc.borrow();

        // Get KV cache buffers only if KV cache exists (LLM mode)
        let has_kv_cache = state.cache_layers().is_some();

        let key_cache_binding = has_kv_cache.then(|| state.arrays(&[ArrayId::Keys(self.layer_index)]));
        let key_cache_array = key_cache_binding.as_ref().map(|b| b[0].borrow_mut());
        let key_cache_buf_rc = key_cache_array.as_ref().map(|a| a.buffer());
        let mut key_cache_buf_borrow = key_cache_buf_rc.as_ref().map(|rc| rc.borrow_mut());

        let value_cache_binding = has_kv_cache.then(|| state.arrays(&[ArrayId::Values(self.layer_index)]));
        let value_cache_array = value_cache_binding.as_ref().map(|b| b[0].borrow_mut());
        let value_cache_buf_rc = value_cache_array.as_ref().map(|a| a.buffer());
        let mut value_cache_buf_borrow = value_cache_buf_rc.as_ref().map(|rc| rc.borrow_mut());

        let extracted_values_binding = (!has_kv_cache).then(|| state.arrays(&[ArrayId::ExtractedValues]));
        let extracted_values_array = extracted_values_binding.as_ref().map(|b| b[0].borrow_mut());
        let extracted_values_buf_rc = extracted_values_array.as_ref().map(|a| a.buffer());
        let mut extracted_values_buf_borrow = extracted_values_buf_rc.as_ref().map(|rc| rc.borrow_mut());

        // For classifiers (no KV cache): extract values from QKV into a dedicated extracted_values buffer.
        if !has_kv_cache {
            self.update_kv_cache_inplace_kernel.encode(
                None::<&B::Buffer>,
                qkv_buf_borrow.deref(),
                // keys already in desired layout; harmless overwrite
                rotated_keys_buf_borrow.deref_mut(),
                extracted_values_buf_borrow.as_mut().unwrap().deref_mut(),
                num_groups as u32,
                num_heads as u32,
                head_dim as u32,
                suffix_length as u32,
                0u32,
                max_sequence_length as u32,
                command_buffer,
            );
        }

        let queries_array = rotated_queries_binding[0].borrow_mut();
        let queries_buf_rc = queries_array.buffer();
        let queries_buf_borrow = queries_buf_rc.borrow();

        let attention_output_array = attention_output_binding[0].borrow_mut();
        let attention_output_buf_rc = attention_output_array.buffer();
        let mut attention_output_buf_borrow = attention_output_buf_rc.borrow_mut();

        let attention_bias_array_borrow = if use_mask {
            Some(
                attention_bias_binding[0]
                    .get(&window_length)
                    .unwrap_or_else(|| panic!("Attention bias buffer not found for window length {:?}", window_length))
                    .borrow(),
            )
        } else {
            None
        };
        let attention_bias_buf_rc = attention_bias_array_borrow.as_ref().map(|a| a.buffer());
        let attention_bias_buf_borrow = attention_bias_buf_rc.as_ref().map(|rc| rc.borrow());
        let attention_bias_buffer: Option<&B::Buffer> = attention_bias_buf_borrow.as_ref().map(|b| b.deref());

        let partials_binding = state.arrays(&[ArrayId::AttentionPartials]);
        let sums_binding = state.arrays(&[ArrayId::AttentionSums]);
        let maxs_binding = state.arrays(&[ArrayId::AttentionMaxs]);

        let partials_array = partials_binding[0].borrow_mut();
        let partials_buf_rc = partials_array.buffer();
        let mut partials_buf_borrow = partials_buf_rc.borrow_mut();

        let sums_array = sums_binding[0].borrow_mut();
        let sums_buf_rc = sums_array.buffer();
        let mut sums_buf_borrow = sums_buf_rc.borrow_mut();

        let maxs_array = maxs_binding[0].borrow_mut();
        let maxs_buf_rc = maxs_array.buffer();
        let mut maxs_buf_borrow = maxs_buf_rc.borrow_mut();

        let sinks_borrow = sinks_binding.as_ref().map(|binding| binding[0].borrow());
        let sinks_buf_rc = sinks_borrow.as_ref().map(|b| b.buffer());
        let sinks_buf_borrow = sinks_buf_rc.as_ref().map(|rc| rc.borrow());
        let sinks_buffer: Option<&B::Buffer> = sinks_buf_borrow.as_ref().map(|b| b.deref());

        // Only update KV cache for LLM mode (not for classifiers)
        if has_kv_cache {
            self.update_kv_cache_kernel.encode(
                Some(rotated_keys_buf_borrow.deref()),
                qkv_buf_borrow.deref(),
                key_cache_buf_borrow.as_mut().unwrap().deref_mut(),
                value_cache_buf_borrow.as_mut().unwrap().deref_mut(),
                num_groups as u32,
                num_heads as u32,
                head_dim as u32,
                suffix_length as u32,
                segment_prefix_length as u32,
                max_sequence_length as u32,
                command_buffer,
            );
        }

        let key_cache_buffer: &B::Buffer = if has_kv_cache {
            key_cache_buf_borrow.as_ref().unwrap().deref()
        } else {
            rotated_keys_buf_borrow.deref()
        };
        let value_cache_buffer: &B::Buffer = if has_kv_cache {
            value_cache_buf_borrow.as_ref().unwrap().deref()
        } else {
            extracted_values_buf_borrow.as_ref().unwrap().deref()
        };

        let k_head_stride = (max_sequence_length * head_dim) as i32;
        let k_seq_stride = head_dim as i32;
        let v_head_stride = (max_sequence_length * head_dim) as i32;
        let v_seq_stride = head_dim as i32;

        let kernel_key = KernelKey {
            head_dim: head_dim as u32,
            has_mask: use_mask,
        };

        let mut mask_kv_seq_stride_opt: Option<u32> = None;
        let mut mask_q_seq_stride_opt: Option<u32> = None;
        let mut mask_head_stride_opt: Option<u32> = None;
        if attention_bias_buffer.is_some() {
            mask_kv_seq_stride_opt = Some(mask_kv_seq_stride as u32);
            mask_q_seq_stride_opt = Some(mask_q_seq_stride as u32);
            mask_head_stride_opt = Some(mask_head_stride as u32);
        }

        match variant {
            KernelVariant::Gemm => {
                let args = AttentionGemmArguments {
                    queries_buffer: queries_buf_borrow.deref(),
                    keys_buffer: key_cache_buffer,
                    values_buffer: value_cache_buffer,
                    output_buffer: attention_output_buf_borrow.deref_mut(),
                    mask_buffer: attention_bias_buffer,
                    sinks_buffer,
                    num_heads,
                    num_groups,
                    suffix_length,
                    sequence_length,
                    segment_prefix_length,
                    max_sequence_length,
                    head_dim,
                    is_causal: self.is_causal,
                    scale,
                };
                self.gemm_block
                    .encode(state.context(), command_buffer, args)
                    .expect("Failed to encode AttentionGemmBlock");
            },
            KernelVariant::SinglePass => {
                let kernel = match self.single_pass_kernels.get(&kernel_key) {
                    Some(k) => k,
                    None => panic!("Can not find AttentionSinglePassKernel for key {:?}", kernel_key),
                };
                kernel.encode(
                    queries_buf_borrow.deref(),
                    key_cache_buffer,
                    value_cache_buffer,
                    attention_output_buf_borrow.deref_mut(),
                    gqa_factor as u32,
                    sequence_length as u32,
                    k_head_stride as u32,
                    k_seq_stride as u32,
                    v_head_stride as u32,
                    v_seq_stride as u32,
                    scale,
                    attention_bias_buffer,
                    mask_kv_seq_stride_opt,
                    mask_q_seq_stride_opt,
                    mask_head_stride_opt,
                    sinks_buffer,
                    num_heads as u32,
                    suffix_length as u32,
                    command_buffer,
                )
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
                kernel_pass1.encode(
                    queries_buf_borrow.deref(),
                    key_cache_buffer,
                    value_cache_buffer,
                    partials_buf_borrow.deref_mut(),
                    sums_buf_borrow.deref_mut(),
                    maxs_buf_borrow.deref_mut(),
                    gqa_factor as u32,
                    sequence_length as u32,
                    k_head_stride as u32,
                    k_seq_stride as u32,
                    v_head_stride as u32,
                    v_seq_stride as u32,
                    scale,
                    num_heads as u32,
                    suffix_length as u32,
                    attention_bias_buffer,
                    mask_kv_seq_stride_opt,
                    mask_q_seq_stride_opt,
                    mask_head_stride_opt,
                    sinks_buffer,
                    command_buffer,
                );
                kernel_pass2.encode(
                    partials_buf_borrow.deref(),
                    sums_buf_borrow.deref(),
                    maxs_buf_borrow.deref(),
                    attention_output_buf_borrow.deref_mut(),
                    num_heads as u32,
                    suffix_length as u32,
                    command_buffer,
                );
            },
        }

        Ok(())
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
    pub has_mask: bool,
}
