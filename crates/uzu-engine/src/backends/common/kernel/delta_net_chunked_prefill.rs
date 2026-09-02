use crate::{
    backends::common::{Allocation, Backend, Encoder, Kernels, kernel::Unsupported},
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
    pub suffix_len: u32,
}

pub trait DeltaNetChunkedPrefill: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<DeltaNetChunkedPrefill = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        outer_data_type: DataType,
        head_dim: u32,
    ) -> Result<Option<Self>, <Self::Backend as Backend>::Error>;

    fn should_use(
        &self,
        suffix_len: u32,
    ) -> bool;

    fn encode(
        &self,
        args: DeltaNetChunkedPrefillArgs<'_, Self::Backend>,
        encoder: &mut Encoder<Self::Backend>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}

impl<B: Backend<Kernels: Kernels<DeltaNetChunkedPrefill = Unsupported<B>>>> DeltaNetChunkedPrefill for Unsupported<B> {
    type Backend = B;

    fn new(
        _context: &B::Context,
        _outer_data_type: DataType,
        _head_dim: u32,
    ) -> Result<Option<Self>, B::Error> {
        Ok(None)
    }

    fn should_use(
        &self,
        _suffix_len: u32,
    ) -> bool {
        match self.never {}
    }

    fn encode(
        &self,
        _args: DeltaNetChunkedPrefillArgs<'_, B>,
        _encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        match self.never {}
    }
}
