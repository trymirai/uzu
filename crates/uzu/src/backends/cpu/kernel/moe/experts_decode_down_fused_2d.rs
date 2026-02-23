use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{BufferArg, MoeExpertsDecodeDownFused2DKernel},
        },
        cpu::backend::Cpu,
    },
};

pub struct MoeExpertsDecodeDownFused2DCpuKernel;

impl MoeExpertsDecodeDownFused2DKernel for MoeExpertsDecodeDownFused2DCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _AccumT: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'hidden, 'row_expert_map, 'w2_all, 'down_biases, 'y_out, 'encoder>(
        &self,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _row_expert_map: impl BufferArg<'row_expert_map, <Self::Backend as Backend>::NativeBuffer>,
        _w2_all: impl BufferArg<'w2_all, <Self::Backend as Backend>::NativeBuffer>,
        _down_biases: impl BufferArg<'down_biases, <Self::Backend as Backend>::NativeBuffer>,
        _y_out: impl BufferArg<'y_out, <Self::Backend as Backend>::NativeBuffer>,
        _total_rows: u32,
        _d_model: u32,
        _d_ff: u32,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<
        'hidden,
        'row_expert_map,
        'w2_all,
        'down_biases,
        'y_out,
        'encoder,
        'predicate,
    >(
        &self,
        _hidden: impl BufferArg<'hidden, <Self::Backend as Backend>::NativeBuffer>,
        _row_expert_map: impl BufferArg<'row_expert_map, <Self::Backend as Backend>::NativeBuffer>,
        _w2_all: impl BufferArg<'w2_all, <Self::Backend as Backend>::NativeBuffer>,
        _down_biases: impl BufferArg<'down_biases, <Self::Backend as Backend>::NativeBuffer>,
        _y_out: impl BufferArg<'y_out, <Self::Backend as Backend>::NativeBuffer>,
        _total_rows: u32,
        _d_model: u32,
        _d_ff: u32,
        _e: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
