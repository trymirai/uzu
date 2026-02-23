use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MaskUpdateKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MaskUpdateCpuKernel;

impl MaskUpdateKernel for MaskUpdateCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'mask, 'encoder>(
        &self,
        _mask: impl BufferArg<'mask, <Self::Backend as Backend>::NativeBuffer>,
        _unmask_col: i32,
        _mask_col: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'mask, 'encoder, 'predicate>(
        &self,
        _mask: impl BufferArg<'mask, <Self::Backend as Backend>::NativeBuffer>,
        _unmask_col: i32,
        _mask_col: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
