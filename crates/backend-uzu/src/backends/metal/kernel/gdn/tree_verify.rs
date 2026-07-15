use crate::{
    backends::{
        common::{
            Allocation, Encoder, Kernels,
            kernel::{
                BuildTreeGramKernel, BuildTreeOutKernel, BuildTreePrefixKernel, TreeUpdateSolveKernel,
                delta_net_tree_verify::DeltaNetTreeVerify,
            },
        },
        metal::{Metal, MetalContext, device_tier::DeviceTier, error::MetalError, kernel::MetalKernels},
    },
    data_type::DataType,
    encodable_block::mixer::delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments, encode_with},
};

pub struct MetalDeltaNetTreeVerify {
    arguments: TreeVerifyNewArguments,
    prefix: <MetalKernels as Kernels>::BuildTreePrefixKernel,
    gram: <MetalKernels as Kernels>::BuildTreeGramKernel,
    solve: <MetalKernels as Kernels>::TreeUpdateSolveKernel,
    out: <MetalKernels as Kernels>::BuildTreeOutKernel,
}

impl DeltaNetTreeVerify<Metal> for MetalDeltaNetTreeVerify {
    fn is_supported(
        arguments: &TreeVerifyNewArguments,
        _context: &MetalContext,
    ) -> Result<bool, MetalError> {
        Ok(arguments.head_k_dim == 128
            && arguments.head_v_dim == 128
            && matches!(arguments.data_type, DataType::F32 | DataType::BF16))
    }

    fn new(
        context: &MetalContext,
        arguments: &TreeVerifyNewArguments,
    ) -> Result<Self, MetalError> {
        let use_mxu = arguments.data_type == DataType::BF16 && context.supports_mxu();
        let transposed_h0 =
            !use_mxu && matches!(context.device_tier(), DeviceTier::SmallLegacy | DeviceTier::SmallApple8);
        Ok(Self {
            arguments: *arguments,
            prefix: <MetalKernels as Kernels>::BuildTreePrefixKernel::new(context)?,
            gram: <MetalKernels as Kernels>::BuildTreeGramKernel::new(context, arguments.data_type, use_mxu, true)?,
            solve: <MetalKernels as Kernels>::TreeUpdateSolveKernel::new(context, arguments.data_type, 32, true)?,
            out: <MetalKernels as Kernels>::BuildTreeOutKernel::new(
                context,
                arguments.data_type,
                arguments.data_type,
                use_mxu,
                transposed_h0,
                true,
            )?,
        })
    }

    fn encode(
        &self,
        arguments: TreeVerifyEncodeArguments<'_, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<Allocation<Metal>, MetalError> {
        encode_with(&self.arguments, &self.prefix, &self.gram, &self.solve, &self.out, arguments, encoder)
    }
}
