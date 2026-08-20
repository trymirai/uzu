use crate::{
    backends::common::{
        Backend, BufferArg, Encoder, Kernels,
        kernel::matmul::{
            arguments::MatmulArguments,
            routing::{A8ActivationPlan, ActivationFormat, MatmulShape},
        },
    },
    data_type::DataType,
};

pub trait MatmulKernel: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<MatmulKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode<'a, 'b, 'd, TB: BufferArg<'b, Self::Backend>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Self::Backend, TB>,
        encoder: &mut Encoder<Self::Backend>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    fn a8_activation_plan(
        &self,
        _candidate: &MatmulShape,
        _context: &<Self::Backend as Backend>::Context,
    ) -> Option<A8ActivationPlan> {
        None
    }

    fn select_activation_format(
        &self,
        _bf16_shape: &MatmulShape,
        _context: &<Self::Backend as Backend>::Context,
    ) -> ActivationFormat {
        ActivationFormat::Bf16
    }

    /// Whether `encode` accepts [super::d_ops::MatmulDOps::gate_act] for
    /// `shape`, fusing the gated activation into the projection and halving
    /// the output width.
    fn supports_fused_gate_act(
        &self,
        _shape: &MatmulShape,
    ) -> bool {
        false
    }
}
