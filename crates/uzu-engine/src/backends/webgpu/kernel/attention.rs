use super::{AttentionSinglePassWebGpuKernel, SigmoidGateWebGpuKernel, shader::Shader};
use crate::backends::{
    common::{
        Backend, BufferRef, CommandBufferEncoding,
        kernel::{
            AttentionArguments, AttentionKernel, AttentionKernelConfig, AttentionSinglePassKernel, SigmoidGateKernel,
        },
    },
    webgpu::{WebGpu, WebGpuError, command_buffer::WebGpuCommandBufferEncoding, context::WebGpuContext},
};

pub struct WebGpuAttention {
    config: AttentionKernelConfig,
}
impl AttentionKernel for WebGpuAttention {
    type Backend = WebGpu;
    fn new(
        _context: &WebGpuContext,
        config: AttentionKernelConfig,
    ) -> Result<Self, WebGpuError> {
        if config.head_dim == 0 || config.head_dim > 256 {
            return Err(WebGpuError::Unsupported(format!("attention head dimension {}", config.head_dim)));
        }
        Ok(Self {
            config,
        })
    }
    fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            WebGpu,
            impl BufferRef<Backend = WebGpu>,
            impl BufferRef<Backend = WebGpu>,
            impl BufferRef<Backend = WebGpu>,
            impl BufferRef<Backend = WebGpu>,
            impl BufferRef<Backend = WebGpu>,
        >,
        command: &mut WebGpuCommandBufferEncoding,
    ) -> Result<<WebGpu as Backend>::ScratchBuffer, WebGpuError> {
        let c = self.config;
        let mut output =
            command.allocate_scratch_for_shape(&[arguments.suffix_length, c.num_q_heads, c.head_dim], c.data_type)?;
        let kernel = AttentionSinglePassWebGpuKernel::new(
            command.context(),
            c.data_type,
            c.head_dim,
            c.has_sinks,
            c.is_kv_cache_ring,
            c.is_causal,
            arguments.trie.is_some(),
            c.sliding_window_size.is_some(),
        )?;
        kernel.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut output,
            c.num_q_heads / c.num_groups,
            arguments.cache.prefix_len() + arguments.suffix_length,
            c.head_dim,
            c.num_groups * c.head_dim,
            c.head_dim,
            c.num_groups * c.head_dim,
            arguments.cache.ring_params(),
            c.scale.unwrap_or(1.0 / (c.head_dim as f32).sqrt()),
            arguments.trie,
            c.sliding_window_size,
            arguments.sinks,
            c.num_q_heads,
            arguments.suffix_length,
            command,
        );
        if let (Some(gate), Some(stride)) = (arguments.gate, c.gate_row_stride) {
            SigmoidGateWebGpuKernel::new(command.context(), c.data_type)?.encode(
                gate,
                &mut output,
                c.num_q_heads * c.head_dim,
                arguments.suffix_length,
                stride,
                command,
            );
        }
        Ok(output)
    }
}

pub(super) fn encode(
    name: &str,
    shader: &mut Shader,
) {
    match name {
        "AttentionPrepare" => prepare(shader),
        "AttentionSinglePass" => attention(shader),
        "KVCacheUpdate" => update(shader),
        _ => unreachable!(),
    }
}

fn prepare(s: &mut Shader) {
    let heads = s.u("num_q_heads")
        + if s.flag("has_kv") {
            2 * s.u("num_kv_heads")
        } else {
            0
        };
    s.set("total_heads", heads);
    s.groups = [(s.u("batch_dim") * heads * s.u("head_dim")).div_ceil(256), 1, 1];
    s.kernel("attention_prepare", 0);
}
fn update(s: &mut Shader) {
    s.groups = [s.u("element_dim").div_ceil(256), 1, 1];
    s.kernel("kv_cache_update", 0);
}
fn attention(s: &mut Shader) {
    let subgroup_width = s.u("__subgroup_width");
    if subgroup_width > 0 {
        s.groups = [s.u("num_heads"), s.u("suffix_length"), 1];
        s.kernel(
            if s.u("suffix_length") == 1 && s.u("sequence_length") >= 16 {
                "attention_parallel"
            } else {
                "attention_subgroup"
            },
            subgroup_width,
        );
        return;
    }
    let width = s.u("HEAD_DIM").next_power_of_two().max(32);
    s.set("lane_count", width);
    s.groups = [s.u("num_heads"), s.u("suffix_length"), 1];
    s.kernel("attention", width);
}
