use crate::{
    DataType,
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, QkUnpackKernel},
    },
};

pub struct QkUnpack<B: Backend> {
    unpack_kernel: <B::Kernels as Kernels>::QkUnpackKernel,
    data_type: DataType,
}

impl<B: Backend> QkUnpack<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            unpack_kernel: <B::Kernels as Kernels>::QkUnpackKernel::new(context, data_type)?,
            data_type,
        })
    }

    pub fn encode(
        &self,
        qkv: &Allocation<B>,
        suffix_length: usize,
        num_heads: usize,
        num_groups: usize,
        head_dim: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Allocation<B>), B::Error> {
        let mut queries =
            encoder.allocate_scratch(size_for_shape(&[num_heads, suffix_length, head_dim], self.data_type))?;
        let mut keys =
            encoder.allocate_scratch(size_for_shape(&[num_groups, suffix_length, head_dim], self.data_type))?;
        self.unpack_kernel.encode(
            qkv,
            &mut queries,
            &mut keys,
            head_dim as u32,
            num_heads as u32,
            num_groups as u32,
            suffix_length as u32,
            encoder,
        );
        Ok((queries, keys))
    }
}
