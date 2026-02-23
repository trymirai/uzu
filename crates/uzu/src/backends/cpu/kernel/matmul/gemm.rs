use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            gpu_types::GEMMParams,
            kernel::{BufferArg, MatmulGemmKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MatmulGemmCpuKernel;

impl MatmulGemmKernel for MatmulGemmCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _block_rows: u32,
        _block_cols: u32,
        _block_depth: u32,
        _warps_per_row: u32,
        _warps_per_col: u32,
        _align_m: bool,
        _align_n: bool,
        _align_k: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'a, 'b, 'd, 'encoder>(
        &self,
        _a: impl BufferArg<'a, <Self::Backend as Backend>::NativeBuffer>,
        _b: impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>,
        _d: impl BufferArg<'d, <Self::Backend as Backend>::NativeBuffer>,
        _params: &[GEMMParams],
        _group_count_x: u32,
        _group_count_y: u32,
        _group_count_z: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'a, 'b, 'd, 'encoder, 'predicate>(
        &self,
        _a: impl BufferArg<'a, <Self::Backend as Backend>::NativeBuffer>,
        _b: impl BufferArg<'b, <Self::Backend as Backend>::NativeBuffer>,
        _d: impl BufferArg<'d, <Self::Backend as Backend>::NativeBuffer>,
        _params: &[GEMMParams],
        _group_count_x: u32,
        _group_count_y: u32,
        _group_count_z: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
