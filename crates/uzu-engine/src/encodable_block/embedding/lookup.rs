use super::resource::EmbeddingStorage;
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels, gpu_types::EmbeddingTableKind,
        kernel::InputEmbeddingLookupKernel,
    },
    encodable_block::{EncodableBlock, embedding::EmbeddingResource},
};

type LookupKernel<B> = <<B as Backend>::Kernels as Kernels>::InputEmbeddingLookupKernel;

pub struct EmbeddingLookupInput<'a, B: Backend, T: BufferRef<Backend = B>> {
    pub resource: &'a EmbeddingResource<B>,
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
    kernel: LookupKernel<B>,
    scale: f32,
}

impl<B: Backend> EmbeddingLookup<B> {
    pub fn new(
        context: &B::Context,
        resource: &EmbeddingResource<B>,
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
            kernel,
            scale,
        })
    }
}

impl<B: Backend, T: BufferRef<Backend = B>> EncodableBlock<B, EmbeddingLookupInput<'_, B, T>> for EmbeddingLookup<B> {
    type Kernel = LookupKernel<B>;
    type Output = B::ScratchBuffer;
    type Error = B::Error;

    fn encode(
        &self,
        input: EmbeddingLookupInput<'_, B, T>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self::Output, Self::Error> {
        command_buffer.push_debug_group("embedding lookup");

        let mut output = command_buffer
            .allocate_scratch_for_shape(&[input.batch_dim, input.resource.model_dim], input.resource.data_type)?;
        let bindings = LookupBindings::new(&input.resource.storage);
        self.kernel.encode(
            input.token_ids,
            bindings.values,
            bindings.scales,
            bindings.zero_points,
            bindings.biases,
            input.resource.output_hadamard_factors.as_ref(),
            bindings.ladder_indices,
            bindings.ladder,
            bindings.codebook,
            &mut output,
            input.batch_dim,
            input.resource.vocab_size,
            input.resource.model_dim,
            self.scale,
            command_buffer,
        );

        command_buffer.pop_debug_group();

        Ok(output)
    }
}
