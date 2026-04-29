use super::*;
use crate::array::{Array, ArrayContextExt};

pub(super) fn structured_audio_dtype_key(data_type: DataType) -> u8 {
    match data_type {
        DataType::F16 => 1,
        DataType::BF16 => 2,
        _ => 0,
    }
}

pub(super) struct StructuredAudioPostModuleRuntime<B: Backend> {
    pub(super) context: Rc<B::Context>,
    pub(super) model_shape: ModelShape,
    pub(super) shared_buffers: Rc<SharedBuffers<B>>,
    pub(super) layers: Box<[LayerExecutables<B>]>,
    pub(super) output_norm: RMSNorm<B>,
    pub(super) max_sequence_length: usize,
}

pub(super) struct StructuredAudioKernelCache<B: Backend> {
    pub(super) half_snake: <B::Kernels as Kernels>::AudioHalfSnakeKernel,
    pub(super) causal_conv1d: <B::Kernels as Kernels>::AudioCausalConv1dKernel,
    pub(super) causal_conv1d_grouped: <B::Kernels as Kernels>::AudioCausalConv1dGroupedKernel,
    pub(super) causal_conv1d_grouped_residual: <B::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel,
    pub(super) causal_conv_transpose1d_causal_pad: <B::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel,
    pub(super) conv1d: <B::Kernels as Kernels>::AudioConv1dKernel,
    pub(super) norm_ncs: <B::Kernels as Kernels>::AudioNormNcsKernel,
    pub(super) activation: <B::Kernels as Kernels>::ActivationKernel,
    pub(super) add: <B::Kernels as Kernels>::AudioAddKernel,
}

pub(super) struct FishAudioQuantizerResources<B: Backend> {
    pub(super) data_type: DataType,
    pub(super) codebook_dim: usize,
    pub(super) residual_quantizers: usize,
    pub(super) semantic_cardinality: usize,
    pub(super) residual_cardinality: usize,
    pub(super) kernel: <B::Kernels as Kernels>::AudioQuantizerDecodeKernel,
    pub(super) semantic_codebook: Array<B>,
    pub(super) semantic_out_proj: Array<B>,
    pub(super) semantic_out_bias: Array<B>,
    pub(super) residual_codebooks: Array<B>,
    pub(super) residual_out_proj: Array<B>,
    pub(super) residual_out_bias: Array<B>,
}

pub(super) struct DecodeWorkspace<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> DecodeWorkspace<B> {
    fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }

    pub(super) fn lengths_array(
        &self,
        encoder: &mut Encoder<B>,
        min_elements: usize,
    ) -> Array<B> {
        let allocation = encoder
            .allocate_constant(min_elements * std::mem::size_of::<i32>())
            .expect("Failed to allocate structured audio lengths");
        unsafe { Array::from_allocation(allocation, 0, &[min_elements], DataType::I32) }
    }

    pub(super) fn next_scratch(
        &self,
        encoder: &mut Encoder<B>,
        shape: &[usize],
        data_type: DataType,
    ) -> Array<B> {
        let allocation =
            encoder.allocate_scratch(size_for_shape(shape, data_type)).expect("Failed to allocate scratch");
        unsafe { Array::from_allocation(allocation, 0, shape, data_type) }
    }

    pub(super) fn reset(&self) {}
}

pub(in crate::audio::nanocodec::runtime) struct StructuredAudioRuntimeResources<B: Backend> {
    context: Rc<B::Context>,
    pub(super) kernels_by_dtype: RefCell<HashMap<u8, Rc<StructuredAudioKernelCache<B>>>>,
    pub(super) post_module_runtime: RefCell<Option<Rc<StructuredAudioPostModuleRuntime<B>>>>,
    pub(super) vocoder_graph: RefCell<Option<Rc<StructuredAudioDecoderGraph<B>>>>,
    pub(super) quantizer_resources: RefCell<Option<Rc<FishAudioQuantizerResources<B>>>>,
    pub(super) decode_workspace: DecodeWorkspace<B>,
}

impl<B: Backend> StructuredAudioRuntimeResources<B> {
    pub(in crate::audio::nanocodec::runtime) fn new(context: Rc<B::Context>) -> Self {
        Self {
            context,
            kernels_by_dtype: RefCell::new(HashMap::new()),
            post_module_runtime: RefCell::new(None),
            vocoder_graph: RefCell::new(None),
            quantizer_resources: RefCell::new(None),
            decode_workspace: DecodeWorkspace::new(),
        }
    }

    pub(super) fn context(&self) -> &Rc<B::Context> {
        &self.context
    }

    /// Reset the decode workspace after submitting a command buffer whose
    /// encoded work references the current buffers.
    pub(in crate::audio::nanocodec::runtime) fn reset_for_pending(&self) {
        self.decode_workspace.reset();
    }

    pub(super) fn kernels(
        &self,
        data_type: DataType,
    ) -> AudioResult<Rc<StructuredAudioKernelCache<B>>> {
        let key = structured_audio_dtype_key(data_type);
        if let Some(existing) = self.kernels_by_dtype.borrow().get(&key) {
            return Ok(existing.clone());
        }
        let created = Rc::new(build_structured_audio_kernels(&self.context, data_type)?);
        self.kernels_by_dtype.borrow_mut().insert(key, created.clone());
        Ok(created)
    }

    pub(super) fn token_staging(
        &self,
        min_elements: usize,
    ) -> Array<B> {
        self.context.create_array_uninitialized(&[min_elements], DataType::U32, "structured_audio_tokens")
    }

    pub(super) fn length_staging(
        &self,
        min_elements: usize,
    ) -> Array<B> {
        self.context.create_array_uninitialized(&[min_elements], DataType::I32, "structured_audio_lengths")
    }
}

pub(super) fn build_structured_audio_kernels<B: Backend>(
    context: &Rc<B::Context>,
    data_type: DataType,
) -> AudioResult<StructuredAudioKernelCache<B>> {
    Ok(StructuredAudioKernelCache {
        half_snake: <B::Kernels as Kernels>::AudioHalfSnakeKernel::new(context.as_ref(), data_type)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize snake1d kernel: {err}")))?,
        causal_conv1d: <B::Kernels as Kernels>::AudioCausalConv1dKernel::new(context.as_ref(), data_type)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize causal conv1d kernel: {err}")))?,
        causal_conv1d_grouped: <B::Kernels as Kernels>::AudioCausalConv1dGroupedKernel::new(
            context.as_ref(),
            data_type,
        )
        .map_err(|err| AudioError::Runtime(format!("failed to initialize grouped causal conv1d kernel: {err}")))?,
        causal_conv1d_grouped_residual: <B::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel::new(
            context.as_ref(),
            data_type,
        )
        .map_err(|err| AudioError::Runtime(format!("failed to initialize grouped residual conv1d kernel: {err}")))?,
        causal_conv_transpose1d_causal_pad: <B::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel::new(
            context.as_ref(),
            data_type,
        )
        .map_err(|err| AudioError::Runtime(format!("failed to initialize causal transpose-conv kernel: {err}")))?,
        conv1d: <B::Kernels as Kernels>::AudioConv1dKernel::new(context.as_ref(), data_type)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize conv1d kernel: {err}")))?,
        norm_ncs: <B::Kernels as Kernels>::AudioNormNcsKernel::new(context.as_ref(), data_type)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize norm kernel: {err}")))?,
        activation: <B::Kernels as Kernels>::ActivationKernel::new(context.as_ref(), data_type, false)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize activation kernel: {err}")))?,
        add: <B::Kernels as Kernels>::AudioAddKernel::new(context.as_ref(), data_type)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize add kernel: {err}")))?,
    })
}
