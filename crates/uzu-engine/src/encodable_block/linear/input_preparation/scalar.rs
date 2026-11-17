use std::sync::Arc;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding,
        gpu_types::hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
        kernel::{
            ActivationTransform,
            activation_transform::SUPPORTED_HADAMARD_BLOCK_SIZES,
            matmul::{ActivationFormat, MatmulA},
        },
    },
    data_type::DataType,
    encodable_block::linear::{LinearBlockError, LinearInput},
    safetensors::ParameterTree,
};

pub struct InputRhtParameters<B: Backend> {
    pub hadamard_size: u32,
    pub signs: Arc<B::GlobalGpuBuffer>,
}

impl<B: Backend> InputRhtParameters<B> {
    pub async fn load(
        input_dim: u32,
        size: u32,
        tree: &ParameterTree<'_, B>,
        loaded: &mut Vec<(Self, Box<[i32]>)>,
    ) -> Result<usize, LinearBlockError<B>> {
        if !SUPPORTED_HADAMARD_BLOCK_SIZES.contains(&size) || !input_dim.is_multiple_of(size) {
            return Err(LinearBlockError::UnsupportedConfiguration(format!(
                "Hadamard block size {size} for input dimension {input_dim}"
            )));
        }
        let signs = tree.read_slice::<i32>("incoherence_signs.input_signs", &[input_dim]).await?;
        if let Some(index) = loaded.iter().position(|(rht, values)| rht.hadamard_size == size && *values == signs) {
            return Ok(index);
        }
        let rht = Self {
            hadamard_size: size,
            signs: Arc::new(tree.read_buffer("incoherence_signs.input_signs", &[input_dim], DataType::I32).await?),
        };
        let index = loaded.len();
        loaded.push((rht, signs));
        Ok(index)
    }

    pub fn h32(signs: Arc<B::GlobalGpuBuffer>) -> Self {
        Self {
            hadamard_size: HADAMARD_TRANSFORM_BLOCK_SIZE,
            signs,
        }
    }
}

pub struct ScalarPreparation<B: Backend> {
    rotation: Option<(Arc<B::GlobalGpuBuffer>, ActivationTransform<B>)>,
    quantizer: Option<ActivationTransform<B>>,
    input_dim: u32,
}

impl<B: Backend> ScalarPreparation<B> {
    pub fn requires_full_precision(&self) -> bool {
        self.rotation.is_some()
    }
    pub fn supports_owned_input(&self) -> bool {
        self.rotation.as_ref().is_none_or(|(_, rht)| rht.is_in_place())
    }
    pub fn is_identity(&self) -> bool {
        self.rotation.is_none() && self.quantizer.is_none()
    }

    pub fn new(
        input_dim: u32,
        data_type: DataType,
        rht: Option<&InputRhtParameters<B>>,
        format: ActivationFormat,
        in_place: bool,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let hadamard_size = rht.as_ref().map_or(0, |rht| rht.hadamard_size);
        let rotation = rht
            .map(|rht| {
                ActivationTransform::input_rht(context, data_type, rht.hadamard_size, in_place)
                    .map(|kernel| (rht.signs.clone(), kernel))
            })
            .transpose()?;
        let quantizer = matches!(format, ActivationFormat::Int8(_) | ActivationFormat::PackedHalf { .. })
            .then(|| ActivationTransform::quantize(context, data_type, hadamard_size, format))
            .transpose()?;
        Ok(Self {
            rotation,
            quantizer,
            input_dim,
        })
    }

    pub fn prepare(
        &self,
        input: impl BufferRef<Backend = B>,
        batch_dim: u32,
        format: ActivationFormat,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Option<LinearInput<B>>, B::Error> {
        if let Some(quantizer) = self.quantizer.as_ref().filter(|kernel| kernel.format() == format) {
            return quantizer
                .prepare_quantized(
                    input,
                    self.rotation.as_ref().map(|(signs, _)| &**signs),
                    batch_dim,
                    self.input_dim,
                    command_buffer,
                )
                .map(Some);
        }
        let Some((signs, rht)) = &self.rotation else {
            return Ok(None);
        };
        let mut transformed = command_buffer.allocate_scratch(input.size())?;
        rht.encode_fp(input, &mut transformed, Some(&**signs), batch_dim, self.input_dim, command_buffer);
        Ok(Some(MatmulA::full_precision(transformed)))
    }

    pub fn prepare_owned(
        &self,
        mut input: B::ScratchBuffer,
        offset: usize,
        batch_dim: u32,
        format: ActivationFormat,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<LinearInput<B>, B::Error> {
        if self.quantizer.as_ref().is_some_and(|kernel| kernel.format() == format)
            && let Some(prepared) = self.prepare((&input).subrange(offset..), batch_dim, format, command_buffer)?
        {
            return Ok(prepared);
        }
        if let Some((signs, rht)) = &self.rotation {
            rht.encode_fp_in_place(
                (&mut input).subrange_mut(offset..),
                Some(&**signs),
                None,
                batch_dim,
                self.input_dim,
                command_buffer,
            );
        }
        Ok(MatmulA::FullPrecision {
            values: input,
            offset,
        })
    }
}
