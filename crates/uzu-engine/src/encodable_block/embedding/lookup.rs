use std::sync::Arc;

use super::resource::EmbeddingStorage;
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels, gpu_types::EmbeddingTableKind,
        kernel::InputEmbeddingLookupKernel,
    },
    encodable_block::{EncodableBlock, embedding::EmbeddingResource},
};

type LookupKernel<B> = <<B as Backend>::Kernels as Kernels>::InputEmbeddingLookupKernel;

pub struct EmbeddingLookupInput<T: BufferRef> {
    pub token_ids: T,
    pub batch_dim: u32,
}

struct LookupBindings<'a, B: Backend> {
    values: &'a B::GlobalBuffer,
    scales: Option<&'a B::GlobalBuffer>,
    zero_points: Option<&'a B::GlobalBuffer>,
    biases: Option<&'a B::GlobalBuffer>,
    ladder_indices: Option<&'a B::GlobalBuffer>,
    ladder: Option<&'a B::GlobalBuffer>,
    codebook: Option<&'a B::GlobalBuffer>,
}

impl<'a, B: Backend> LookupBindings<'a, B> {
    fn new(storage: &'a EmbeddingStorage<B>) -> Self {
        match storage {
            EmbeddingStorage::Matrix(matrix) => Self {
                values: matrix.values(),
                scales: matrix.scales(),
                zero_points: matrix.zero_points(),
                biases: matrix.biases(),
                ladder_indices: None,
                ladder: None,
                codebook: None,
            },
            EmbeddingStorage::D4S4(table) => Self {
                values: &table.codes,
                scales: Some(&table.row_scales),
                zero_points: None,
                biases: None,
                ladder_indices: Some(&table.ladder_indices),
                ladder: Some(&table.ladder),
                codebook: Some(&table.codebook),
            },
        }
    }
}

pub struct EmbeddingLookup<B: Backend> {
    resource: Arc<EmbeddingResource<B>>,
    kernel: LookupKernel<B>,
    scale: f32,
}

impl<B: Backend> EmbeddingLookup<B> {
    pub fn new(
        context: &B::Context,
        resource: Arc<EmbeddingResource<B>>,
        scale: f32,
    ) -> Result<Self, B::Error> {
        let (table_kind, quantization) = match &resource.storage {
            EmbeddingStorage::D4S4(_) => (EmbeddingTableKind::D4S4, None),
            EmbeddingStorage::Matrix(matrix) => match matrix.quantization() {
                Some(info) => (EmbeddingTableKind::Quantized, Some(info)),
                None => (EmbeddingTableKind::Dense, None),
            },
        };
        let kernel = LookupKernel::<B>::new(
            context,
            resource.data_type,
            table_kind,
            quantization.map(|info| info.group_size),
            quantization.map(|info| info.mode),
            quantization.map(|info| info.method),
            resource.output_hadamard_factors.is_some(),
        )?;

        Ok(Self {
            resource,
            kernel,
            scale,
        })
    }

    pub fn model_dim(&self) -> u32 {
        self.resource.model_dim
    }
}

impl<B: Backend, T: BufferRef<Backend = B>> EncodableBlock<B, EmbeddingLookupInput<T>> for EmbeddingLookup<B> {
    type Kernel = LookupKernel<B>;
    type Output = B::ScratchBuffer;
    type Error = B::Error;

    fn encode(
        &self,
        input: EmbeddingLookupInput<T>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self::Output, Self::Error> {
        command_buffer.push_debug_group("embedding lookup");

        let mut output = command_buffer
            .allocate_scratch_for_shape(&[input.batch_dim, self.resource.model_dim], self.resource.data_type)?;
        let bindings = LookupBindings::new(&self.resource.storage);
        self.kernel.encode(
            input.token_ids,
            bindings.values,
            bindings.scales,
            bindings.zero_points,
            bindings.biases,
            self.resource.output_hadamard_factors.as_ref(),
            bindings.ladder_indices,
            bindings.ladder,
            bindings.codebook,
            &mut output,
            input.batch_dim,
            self.resource.vocab_size,
            self.resource.model_dim,
            self.scale,
            command_buffer,
        );

        command_buffer.pop_debug_group();

        Ok(output)
    }
}
