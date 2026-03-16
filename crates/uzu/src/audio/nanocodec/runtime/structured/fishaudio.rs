use super::*;
use crate::backends::metal::Metal;

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
    pub(super) scratch_buffers: ScratchBuffers<B>,
    pub(super) shared_buffers: Rc<RefCell<SharedBuffers<B>>>,
    pub(super) layers: Box<[LayerExecutables<B>]>,
    pub(super) output_norm: RMSNorm<B>,
    pub(super) max_sequence_length: usize,
}

pub(super) struct StructuredAudioKernelCache<B: Backend> {
    pub(super) half_snake: <B::Kernels as Kernels>::AudioHalfSnakeKernel,
    pub(super) causal_conv1d: <B::Kernels as Kernels>::AudioCausalConv1dKernel,
    pub(super) causal_conv1d_grouped: <B::Kernels as Kernels>::AudioCausalConv1dGroupedKernel,
    pub(super) causal_conv1d_grouped_residual: <B::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel,
    pub(super) causal_conv_transpose1d_causal_pad:
        <B::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel,
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

pub(super) struct StructuredAudioRuntimeResources<B: Backend> {
    context: Rc<B::Context>,
    pub(super) kernels_by_dtype: RefCell<HashMap<u8, Rc<StructuredAudioKernelCache<B>>>>,
    pub(super) post_module_runtime: RefCell<Option<Rc<StructuredAudioPostModuleRuntime<B>>>>,
    pub(super) vocoder_graph: RefCell<Option<Rc<StructuredAudioDecoderGraph<B>>>>,
    pub(super) quantizer_resources: RefCell<Option<Rc<FishAudioQuantizerResources<B>>>>,
}

impl<B: Backend> StructuredAudioRuntimeResources<B> {
    pub(super) fn new(context: Rc<B::Context>) -> Self {
        Self {
            context,
            kernels_by_dtype: RefCell::new(HashMap::new()),
            post_module_runtime: RefCell::new(None),
            vocoder_graph: RefCell::new(None),
            quantizer_resources: RefCell::new(None),
        }
    }

    pub(super) fn context(&self) -> &Rc<B::Context> {
        &self.context
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
        causal_conv1d_grouped: <B::Kernels as Kernels>::AudioCausalConv1dGroupedKernel::new(context.as_ref(), data_type)
            .map_err(|err| AudioError::Runtime(format!("failed to initialize grouped causal conv1d kernel: {err}")))?,
        causal_conv1d_grouped_residual:
            <B::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel::new(context.as_ref(), data_type).map_err(
                |err| AudioError::Runtime(format!("failed to initialize grouped residual conv1d kernel: {err}")),
            )?,
        causal_conv_transpose1d_causal_pad:
            <B::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel::new(context.as_ref(), data_type)
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

thread_local! {
    pub(super) static FISHAUDIO_RUNTIME_RESOURCES_CACHE: RefCell<HashMap<String, Rc<StructuredAudioRuntimeResources<Metal>>>> =
        RefCell::new(HashMap::new());
}

pub(super) fn structured_audio_runtime_resources(
    weights_path: &str,
) -> AudioResult<Rc<StructuredAudioRuntimeResources<Metal>>> {
    FISHAUDIO_RUNTIME_RESOURCES_CACHE.with(|cache| {
        if let Some(existing) = cache.borrow().get(weights_path) {
            return Ok(existing.clone());
        }
        let context = <Metal as Backend>::Context::new()
            .map_err(|err| AudioError::Runtime(format!("failed to create structured audio decode context: {err}")))?;
        let created = Rc::new(StructuredAudioRuntimeResources::new(context));
        cache.borrow_mut().insert(weights_path.to_string(), created.clone());
        Ok(created)
    })
}
