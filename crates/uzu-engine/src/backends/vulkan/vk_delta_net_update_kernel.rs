use super::{DeltaNetUpdateVulkanKernel, Error, VkBuffer, VkCommandBufferEncoding, VkContext, Vulkan};
use crate::{
    backends::common::{BufferMut, BufferRef, kernel::DeltaNetUpdateKernel},
    data_type::DataType,
};

/// Metal's 512-invocation workgroup, 16 simdgroups of 32 lanes.
const METAL_THREADS: u32 = 512;
/// Vulkan's guaranteed workgroup size, 4 simdgroups.
const MIN_THREADS: u32 = 128;

/// DeltaNetUpdate in Metal's 512-invocation workgroup where the device's limits allow it, otherwise in 128.
pub struct VkDeltaNetUpdateKernel {
    kernel: DeltaNetUpdateVulkanKernel,
}

impl DeltaNetUpdateKernel for VkDeltaNetUpdateKernel {
    type Backend = Vulkan;

    fn new(
        context: &VkContext,
        data_type: DataType,
        head_k_dim: u32,
    ) -> Result<Self, Error> {
        let limits = &context.physical_device().properties.limits;
        let threads = if METAL_THREADS <= limits.max_compute_work_group_invocations
            && METAL_THREADS <= limits.max_compute_work_group_size[0]
        {
            METAL_THREADS
        } else {
            MIN_THREADS
        };
        Ok(Self {
            kernel: DeltaNetUpdateVulkanKernel::new(&context.arc(), data_type, head_k_dim, threads)?,
        })
    }

    fn encode(
        &self,
        in_proj: impl BufferRef<Backend = Vulkan>,
        a_log: impl BufferRef<Backend = Vulkan>,
        dt_bias: impl BufferRef<Backend = Vulkan>,
        norm_weight: impl BufferRef<Backend = Vulkan>,
        state: impl BufferMut<Backend = Vulkan>,
        out: impl BufferMut<Backend = Vulkan>,
        num_v_heads: u32,
        num_k_heads: u32,
        head_v_dim: u32,
        key_dim: u32,
        value_dim: u32,
        norm_epsilon: f32,
        command_buffer: &mut VkCommandBufferEncoding,
    ) {
        let (in_proj, in_proj_range) = in_proj.parts();
        let (a_log, a_log_range) = a_log.parts();
        let (dt_bias, dt_bias_range) = dt_bias.parts();
        let (norm_weight, norm_weight_range) = norm_weight.parts();
        let (state, state_range) = state.parts();
        let (out, out_range) = out.parts();
        // SAFETY: as for the CPU kernel, the common contract makes callers pass ranges that cover every element the
        // kernel indexes, the state and out aliasing no other argument.
        unsafe {
            self.kernel.encode(
                VkBuffer::resolve(in_proj, in_proj_range),
                VkBuffer::resolve(a_log, a_log_range),
                VkBuffer::resolve(dt_bias, dt_bias_range),
                VkBuffer::resolve(norm_weight, norm_weight_range),
                VkBuffer::resolve(state, state_range),
                VkBuffer::resolve(out, out_range),
                num_v_heads,
                num_k_heads,
                head_v_dim,
                key_dim,
                value_dim,
                norm_epsilon,
                command_buffer,
            );
        }
    }
}
