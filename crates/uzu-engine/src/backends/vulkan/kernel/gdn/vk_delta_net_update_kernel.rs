use crate::{
    backends::{
        common::{
            BufferMut, BufferRef,
            gpu_types::activation_type::ActivationType,
            kernel::{DeltaNetNormGateKernel, DeltaNetUpdateKernel},
        },
        vulkan::{
            DeltaNetNormGateVulkanKernel, DeltaNetUpdateVulkanKernel, Error, VkBuffer, VkCommandBufferEncoding,
            VkContext, Vulkan,
        },
    },
    data_type::DataType,
};

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
        act_type: ActivationType,
    ) -> Result<Self, Error> {
        Ok(Self {
            update: DeltaNetUpdateVulkanKernel::new(context, data_type, head_k_dim)?,
            norm_gate: DeltaNetNormGateVulkanKernel::new(context, data_type, act_type)?,
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
        activation_alpha: f32,
        command_buffer: &mut VkCommandBufferEncoding,
    ) {
        let (state, state_range) = state.parts();
        let (output, output_range) = out.reborrow().parts();
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
            activation_alpha,
            1,
            command_buffer,
        );
    }
}
