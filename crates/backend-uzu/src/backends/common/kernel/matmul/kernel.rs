use super::{arguments::MatmulArguments, quant_combo::MatmulQuantCombo};
use crate::{
    DataType,
    backends::common::{AsBufferRangeRef, Backend, Buffer, Encoder, kernel::ManualKernels},
};

pub trait MatmulKernel: Sized {
    type Backend: Backend<Kernels: ManualKernels<MatmulKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Self::Backend>>>(
        &mut self,
        arguments: MatmulArguments<Self::Backend, TB>,
        encoder: &mut Encoder<Self::Backend>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    fn preheat_quant_combo(
        &mut self,
        _context: &<Self::Backend as Backend>::Context,
        _combo: MatmulQuantCombo,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        Ok(())
    }
}
