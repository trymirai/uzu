use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{
                BufferArg, QuantizedMatmulQmmKernel, QuantizedMatmulQmmTransposed64x64Kernel,
                QuantizedMatmulQmmTransposedKernel, QuantizedMatmulQmvFastKernel,
                QuantizedMatmulQmvKernel, QuantizedMatmulQvmKernel,
            },
        },
        cpu::backend::Cpu,
    },
};

pub struct QuantizedMatmulQmmCpuKernel;

impl QuantizedMatmulQmmKernel for QuantizedMatmulQmmCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _group_size: i32,
        _bits: i32,
        _use_zero_points: bool,
        _use_mlx_quant: bool,
        _aligned_k: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'w, 'scales, 'zero_points, 'biases, 'x, 'y, 'encoder>(
        &self,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _zero_points: Option<impl BufferArg<'zero_points, <Self::Backend as Backend>::NativeBuffer>>,
        _biases: Option<impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _k: i32,
        _n: i32,
        _m: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'w, 'scales, 'zero_points, 'biases, 'x, 'y, 'encoder, 'predicate>(
        &self,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _zero_points: Option<impl BufferArg<'zero_points, <Self::Backend as Backend>::NativeBuffer>>,
        _biases: Option<impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _k: i32,
        _n: i32,
        _m: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct QuantizedMatmulQmmTransposedCpuKernel;

impl QuantizedMatmulQmmTransposedKernel for QuantizedMatmulQmmTransposedCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _group_size: i32,
        _bits: i32,
        _use_zero_points: bool,
        _use_mlx_quant: bool,
        _aligned_n: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'w, 'scales, 'zero_points, 'biases, 'x, 'y, 'encoder>(
        &self,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _zero_points: Option<impl BufferArg<'zero_points, <Self::Backend as Backend>::NativeBuffer>>,
        _biases: Option<impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _k: i32,
        _n: i32,
        _m: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'w, 'scales, 'zero_points, 'biases, 'x, 'y, 'encoder, 'predicate>(
        &self,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _zero_points: Option<impl BufferArg<'zero_points, <Self::Backend as Backend>::NativeBuffer>>,
        _biases: Option<impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _k: i32,
        _n: i32,
        _m: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct QuantizedMatmulQmmTransposed64x64CpuKernel;

impl QuantizedMatmulQmmTransposed64x64Kernel for QuantizedMatmulQmmTransposed64x64CpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _group_size: i32,
        _bits: i32,
        _use_zero_points: bool,
        _use_mlx_quant: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'w, 'scales, 'zero_points, 'biases, 'x, 'y, 'encoder>(
        &self,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _zero_points: Option<impl BufferArg<'zero_points, <Self::Backend as Backend>::NativeBuffer>>,
        _biases: Option<impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _k: i32,
        _n: i32,
        _m: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'w, 'scales, 'zero_points, 'biases, 'x, 'y, 'encoder, 'predicate>(
        &self,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _zero_points: Option<impl BufferArg<'zero_points, <Self::Backend as Backend>::NativeBuffer>>,
        _biases: Option<impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _k: i32,
        _n: i32,
        _m: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct QuantizedMatmulQmvCpuKernel;

impl QuantizedMatmulQmvKernel for QuantizedMatmulQmvCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _group_size: i32,
        _bits: i32,
        _use_zero_points: bool,
        _use_mlx_quant: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'w, 'scales, 'zero_points, 'biases, 'x, 'y, 'encoder>(
        &self,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _zero_points: Option<impl BufferArg<'zero_points, <Self::Backend as Backend>::NativeBuffer>>,
        _biases: Option<impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _k: i32,
        _n: i32,
        _m: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'w, 'scales, 'zero_points, 'biases, 'x, 'y, 'encoder, 'predicate>(
        &self,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _zero_points: Option<impl BufferArg<'zero_points, <Self::Backend as Backend>::NativeBuffer>>,
        _biases: Option<impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _k: i32,
        _n: i32,
        _m: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct QuantizedMatmulQmvFastCpuKernel;

impl QuantizedMatmulQmvFastKernel for QuantizedMatmulQmvFastCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _group_size: i32,
        _bits: i32,
        _use_zero_points: bool,
        _use_mlx_quant: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'w, 'scales, 'zero_points, 'biases, 'x, 'y, 'encoder>(
        &self,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _zero_points: Option<impl BufferArg<'zero_points, <Self::Backend as Backend>::NativeBuffer>>,
        _biases: Option<impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _k: i32,
        _n: i32,
        _m: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'w, 'scales, 'zero_points, 'biases, 'x, 'y, 'encoder, 'predicate>(
        &self,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _zero_points: Option<impl BufferArg<'zero_points, <Self::Backend as Backend>::NativeBuffer>>,
        _biases: Option<impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _k: i32,
        _n: i32,
        _m: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct QuantizedMatmulQvmCpuKernel;

impl QuantizedMatmulQvmKernel for QuantizedMatmulQvmCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
        _group_size: i32,
        _bits: i32,
        _use_zero_points: bool,
        _use_mlx_quant: bool,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'w, 'scales, 'zero_points, 'biases, 'x, 'y, 'encoder>(
        &self,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _zero_points: Option<impl BufferArg<'zero_points, <Self::Backend as Backend>::NativeBuffer>>,
        _biases: Option<impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _k: i32,
        _n: i32,
        _m: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'w, 'scales, 'zero_points, 'biases, 'x, 'y, 'encoder, 'predicate>(
        &self,
        _w: impl BufferArg<'w, <Self::Backend as Backend>::NativeBuffer>,
        _scales: impl BufferArg<'scales, <Self::Backend as Backend>::NativeBuffer>,
        _zero_points: Option<impl BufferArg<'zero_points, <Self::Backend as Backend>::NativeBuffer>>,
        _biases: Option<impl BufferArg<'biases, <Self::Backend as Backend>::NativeBuffer>>,
        _x: impl BufferArg<'x, <Self::Backend as Backend>::NativeBuffer>,
        _y: impl BufferArg<'y, <Self::Backend as Backend>::NativeBuffer>,
        _k: i32,
        _n: i32,
        _m: i32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
