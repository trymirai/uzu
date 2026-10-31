use crate::{
    backends::{
        common::{
            BufferMut, BufferRef,
            kernel::{DeltaNetNormGateKernel, DeltaNetUpdateKernel},
        },
        vulkan::{
            DeltaNetNormGateVulkanKernel, DeltaNetUpdateVulkanKernel, Error, VkBuffer, VkCommandBufferEncoding,
            VkContext, Vulkan,
        },
    },
    data_type::DataType,
};

/// DeltaNetUpdate over (head, 16-row) workgroups for parallelism at one token, then DeltaNetNormGate in place for the
/// head-wide RMS and the gate, as the prefill path applies them.
pub struct VkDeltaNetUpdateKernel {
    update: DeltaNetUpdateVulkanKernel,
    norm_gate: DeltaNetNormGateVulkanKernel,
}

impl DeltaNetUpdateKernel for VkDeltaNetUpdateKernel {
    type Backend = Vulkan;

    fn new(
        context: &VkContext,
        data_type: DataType,
        head_k_dim: u32,
    ) -> Result<Self, Error> {
        Ok(Self {
            update: DeltaNetUpdateVulkanKernel::new(&context.arc(), data_type, head_k_dim)?,
            norm_gate: DeltaNetNormGateVulkanKernel::new(context, data_type)?,
        })
    }

    fn encode(
        &self,
        in_proj: impl BufferRef<Backend = Vulkan>,
        a_log: impl BufferRef<Backend = Vulkan>,
        dt_bias: impl BufferRef<Backend = Vulkan>,
        norm_weight: impl BufferRef<Backend = Vulkan>,
        state: impl BufferMut<Backend = Vulkan>,
        mut out: impl BufferMut<Backend = Vulkan>,
        num_v_heads: u32,
        num_k_heads: u32,
        head_v_dim: u32,
        key_dim: u32,
        value_dim: u32,
        norm_epsilon: f32,
        command_buffer: &mut VkCommandBufferEncoding,
    ) {
        let (state, state_range) = state.parts();
        let (output, output_range) = out.reborrow().parts();
        // SAFETY: as for the CPU kernel, the common contract makes callers pass ranges that cover every element the
        // kernel indexes, the state and out aliasing no other argument.
        unsafe {
            self.update.encode(
                VkBuffer::resolve_ref(in_proj),
                VkBuffer::resolve_ref(a_log),
                VkBuffer::resolve_ref(dt_bias),
                VkBuffer::resolve(state, state_range),
                VkBuffer::resolve(output, output_range),
                num_v_heads,
                num_k_heads,
                head_v_dim,
                key_dim,
                value_dim,
                command_buffer,
            );
        }
        let conv_dim = 2 * key_dim + value_dim;
        self.norm_gate.encode(
            out,
            in_proj,
            norm_weight,
            num_v_heads,
            head_v_dim,
            value_dim,
            conv_dim,
            conv_dim + value_dim + 2 * num_v_heads,
            norm_epsilon,
            1,
            command_buffer,
        );
    }
}
