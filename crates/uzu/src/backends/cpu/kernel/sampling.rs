use crate::{
    DataType,
    backends::{
        common::{
            Backend,
            kernel::{
                ArgmaxFinalKernel, ArgmaxMainKernel, ArgmaxSingleKernel, BitmaskKernel,
                BufferArg, GumbelKernel, MinPKernel, TemperatureKernel, TopKKernel, TopPKernel,
            },
        },
        cpu::backend::Cpu,
    },
};

pub struct ArgmaxSingleCpuKernel;

impl ArgmaxSingleKernel for ArgmaxSingleCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'logits_data, 'final_tokens, 'encoder>(
        &self,
        _logits_data: impl BufferArg<'logits_data, <Self::Backend as Backend>::NativeBuffer>,
        _final_tokens: impl BufferArg<'final_tokens, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'logits_data, 'final_tokens, 'encoder, 'predicate>(
        &self,
        _logits_data: impl BufferArg<'logits_data, <Self::Backend as Backend>::NativeBuffer>,
        _final_tokens: impl BufferArg<'final_tokens, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct ArgmaxMainCpuKernel;

impl ArgmaxMainKernel for ArgmaxMainCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'logits_data, 'partial_results, 'encoder>(
        &self,
        _logits_data: impl BufferArg<'logits_data, <Self::Backend as Backend>::NativeBuffer>,
        _partial_results: impl BufferArg<'partial_results, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'logits_data, 'partial_results, 'encoder, 'predicate>(
        &self,
        _logits_data: impl BufferArg<'logits_data, <Self::Backend as Backend>::NativeBuffer>,
        _partial_results: impl BufferArg<'partial_results, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct ArgmaxFinalCpuKernel;

impl ArgmaxFinalKernel for ArgmaxFinalCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'partial_results, 'final_tokens, 'encoder>(
        &self,
        _partial_results: impl BufferArg<'partial_results, <Self::Backend as Backend>::NativeBuffer>,
        _final_tokens: impl BufferArg<'final_tokens, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'partial_results, 'final_tokens, 'encoder, 'predicate>(
        &self,
        _partial_results: impl BufferArg<'partial_results, <Self::Backend as Backend>::NativeBuffer>,
        _final_tokens: impl BufferArg<'final_tokens, <Self::Backend as Backend>::NativeBuffer>,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct BitmaskCpuKernel;

impl BitmaskKernel for BitmaskCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'logits, 'bitmask, 'processed_logits, 'encoder>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _bitmask: impl BufferArg<'bitmask, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'logits, 'bitmask, 'processed_logits, 'encoder, 'predicate>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _bitmask: impl BufferArg<'bitmask, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct GumbelCpuKernel;

impl GumbelKernel for GumbelCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'logits, 'batch_seeds, 'processed_logits, 'encoder>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _batch_seeds: impl BufferArg<'batch_seeds, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'logits, 'batch_seeds, 'processed_logits, 'encoder, 'predicate>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _batch_seeds: impl BufferArg<'batch_seeds, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct MinPCpuKernel;

impl MinPKernel for MinPCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'logits, 'processed_logits, 'encoder>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _min_p: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'logits, 'processed_logits, 'encoder, 'predicate>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _min_p: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct TemperatureCpuKernel;

impl TemperatureKernel for TemperatureCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'logits, 'processed_logits, 'encoder>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _temperature: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'logits, 'processed_logits, 'encoder, 'predicate>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _temperature: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct TopKCpuKernel;

impl TopKKernel for TopKCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'logits, 'processed_logits, 'encoder>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _top_k: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'logits, 'processed_logits, 'encoder, 'predicate>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _top_k: u32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}

pub struct TopPCpuKernel;

impl TopPKernel for TopPCpuKernel {
    type Backend = Cpu;

    fn new(
        _context: &<Self::Backend as Backend>::Context,
        _T: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn encode<'logits, 'processed_logits, 'encoder>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _top_p: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
    ) {
        todo!()
    }

    fn encode_if<'logits, 'processed_logits, 'encoder, 'predicate>(
        &self,
        _logits: impl BufferArg<'logits, <Self::Backend as Backend>::NativeBuffer>,
        _processed_logits: impl BufferArg<
            'processed_logits,
            <Self::Backend as Backend>::NativeBuffer,
        >,
        _batch_size: u32,
        _vocab_size: u32,
        _top_p: f32,
        _encoder: &'encoder <Self::Backend as Backend>::ComputeEncoder,
        _predicate: Option<impl BufferArg<'predicate, <Self::Backend as Backend>::NativeBuffer>>,
    ) {
        todo!()
    }
}
