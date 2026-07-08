use std::convert::Infallible;

use crate::{
    backends::common::{Allocation, Backend, Encoder, Kernels},
    data_type::DataType,
};

pub struct DeltaNetChunkedPrefillArgs<'a, B: Backend> {
    pub in_projected: &'a Allocation<B>,
    pub a_log: &'a Allocation<B>,
    pub dt_bias: &'a Allocation<B>,
    pub ssm_state: &'a mut Allocation<B>,
    pub delta_output: &'a mut Allocation<B>,
    pub num_heads: u32,
    pub num_groups: u32,
    pub value_head_dim: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    pub suffix_len: usize,
}

pub trait DeltaNetChunkedPrefill<B: Backend<Kernels: Kernels<DeltaNetChunkedPrefill = Self>>>: Sized {
    fn new(
        context: &B::Context,
        outer_data_type: DataType,
        head_dim: u32,
    ) -> Result<Option<Self>, B::Error>;

    fn should_use(
        &self,
        suffix_len: usize,
    ) -> bool;

    fn encode(
        &self,
        args: DeltaNetChunkedPrefillArgs<'_, B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error>;
}

impl<B: Backend<Kernels: Kernels<DeltaNetChunkedPrefill = Infallible>>> DeltaNetChunkedPrefill<B> for Infallible {
    fn new(
        _context: &B::Context,
        _outer_data_type: DataType,
        _head_dim: u32,
    ) -> Result<Option<Self>, B::Error> {
        Ok(None)
    }

    fn should_use(
        &self,
        _suffix_len: usize,
    ) -> bool {
        match *self {}
    }

    fn encode(
        &self,
        _args: DeltaNetChunkedPrefillArgs<'_, B>,
        _encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        match *self {}
    }
}
