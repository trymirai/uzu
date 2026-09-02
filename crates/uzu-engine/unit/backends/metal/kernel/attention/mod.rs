#![cfg(backend = "metal")]

use super::{gemm, single_pass, two_pass};
use crate::{array::ArrayElement, backends::common::kernel::AttentionKernelConfig};

mod gemm_test;
mod kernel_test;

fn default_attention_config<T: ArrayElement>(
    head_dim: usize,
    num_q_heads: usize,
    num_groups: usize,
    is_causal: bool,
) -> AttentionKernelConfig {
    AttentionKernelConfig {
        head_dim: head_dim as u32,
        num_groups: num_groups as u32,
        num_q_heads: num_q_heads as u32,
        has_sinks: false,
        is_kv_cache_ring: false,
        is_causal,
        sliding_window_size: None,
        scale: Some(1.0 / (head_dim as f32).sqrt()),
        data_type: T::data_type(),
    }
}
