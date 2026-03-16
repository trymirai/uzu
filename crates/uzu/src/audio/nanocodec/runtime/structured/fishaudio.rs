use super::*;

pub(super) fn structured_audio_dtype_key(data_type: DataType) -> u8 {
    match data_type {
        DataType::F16 => 1,
        DataType::BF16 => 2,
        _ => 0,
    }
}

pub(super) struct StructuredAudioPostModuleRuntime {
    pub(super) context: Rc<<Metal as Backend>::Context>,
    pub(super) model_shape: ModelShape,
    pub(super) scratch_buffers: ScratchBuffers<Metal>,
    pub(super) shared_buffers: Rc<RefCell<SharedBuffers<Metal>>>,
    pub(super) layers: Box<[LayerExecutables<Metal>]>,
    pub(super) output_norm: RMSNorm<Metal>,
    pub(super) max_sequence_length: usize,
}

pub(super) struct StructuredAudioKernelCache {
    pub(super) half_snake: <<Metal as Backend>::Kernels as Kernels>::AudioHalfSnakeKernel,
    pub(super) causal_conv1d: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dKernel,
    pub(super) causal_conv1d_grouped: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedKernel,
    pub(super) causal_conv1d_grouped_residual: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel,
    pub(super) causal_conv_transpose1d_causal_pad:
        <<Metal as Backend>::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel,
    pub(super) conv1d: <<Metal as Backend>::Kernels as Kernels>::AudioConv1dKernel,
    pub(super) norm_ncs: <<Metal as Backend>::Kernels as Kernels>::AudioNormNcsKernel,
    pub(super) activation: <<Metal as Backend>::Kernels as Kernels>::ActivationKernel,
    pub(super) add: <<Metal as Backend>::Kernels as Kernels>::AudioAddKernel,
}

pub(super) struct FishAudioQuantizerResources {
    pub(super) data_type: DataType,
    pub(super) codebook_dim: usize,
    pub(super) residual_quantizers: usize,
    pub(super) semantic_cardinality: usize,
    pub(super) residual_cardinality: usize,
    pub(super) kernel: <<Metal as Backend>::Kernels as Kernels>::AudioQuantizerDecodeKernel,
    pub(super) semantic_codebook: Array<Metal>,
    pub(super) semantic_out_proj: Array<Metal>,
    pub(super) semantic_out_bias: Array<Metal>,
    pub(super) residual_codebooks: Array<Metal>,
    pub(super) residual_out_proj: Array<Metal>,
    pub(super) residual_out_bias: Array<Metal>,
}

thread_local! {
    pub(super) static FISHAUDIO_POST_MODULE_RUNTIME_CACHE: RefCell<HashMap<String, Rc<StructuredAudioPostModuleRuntime>>> =
        RefCell::new(HashMap::new());
    pub(super) static FISHAUDIO_DECODE_CONTEXT_CACHE: RefCell<HashMap<String, Rc<<Metal as Backend>::Context>>> =
        RefCell::new(HashMap::new());
    pub(super) static FISHAUDIO_KERNEL_CACHE: RefCell<HashMap<usize, Rc<StructuredAudioKernelCache>>> =
        RefCell::new(HashMap::new());
    pub(super) static FISHAUDIO_VOCODER_GRAPH_CACHE: RefCell<HashMap<usize, Rc<StructuredAudioDecoderGraph>>> =
        RefCell::new(HashMap::new());
    pub(super) static FISHAUDIO_QUANTIZER_RESOURCES_CACHE: RefCell<HashMap<usize, Rc<FishAudioQuantizerResources>>> =
        RefCell::new(HashMap::new());
}

pub(super) fn structured_audio_kernels(
    context: &Rc<<Metal as Backend>::Context>,
    data_type: DataType,
) -> AudioResult<Rc<StructuredAudioKernelCache>> {
    let key = (Rc::as_ptr(context) as usize).wrapping_mul(31) ^ usize::from(structured_audio_dtype_key(data_type));
    FISHAUDIO_KERNEL_CACHE.with(|cache| {
        if let Some(existing) = cache.borrow().get(&key) {
            return Ok(existing.clone());
        }

        let created = Rc::new(StructuredAudioKernelCache {
            half_snake: <<Metal as Backend>::Kernels as Kernels>::AudioHalfSnakeKernel::new(
                context.as_ref(),
                data_type,
            )
            .map_err(|err| AudioError::Runtime(format!("failed to initialize snake1d kernel: {err}")))?,
            causal_conv1d: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dKernel::new(
                context.as_ref(),
                data_type,
            )
            .map_err(|err| AudioError::Runtime(format!("failed to initialize causal conv1d kernel: {err}")))?,
            causal_conv1d_grouped: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedKernel::new(
                context.as_ref(),
                data_type,
            )
            .map_err(|err| AudioError::Runtime(format!("failed to initialize grouped causal conv1d kernel: {err}")))?,
            causal_conv1d_grouped_residual:
                <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel::new(
                    context.as_ref(),
                    data_type,
                )
                .map_err(|err| {
                    AudioError::Runtime(format!("failed to initialize grouped residual conv1d kernel: {err}"))
                })?,
            causal_conv_transpose1d_causal_pad:
                <<Metal as Backend>::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel::new(
                    context.as_ref(),
                    data_type,
                )
                .map_err(|err| {
                    AudioError::Runtime(format!("failed to initialize causal transpose-conv kernel: {err}"))
                })?,
            conv1d: <<Metal as Backend>::Kernels as Kernels>::AudioConv1dKernel::new(context.as_ref(), data_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize conv1d kernel: {err}")))?,
            norm_ncs: <<Metal as Backend>::Kernels as Kernels>::AudioNormNcsKernel::new(context.as_ref(), data_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize norm kernel: {err}")))?,
            activation: <<Metal as Backend>::Kernels as Kernels>::ActivationKernel::new(
                context.as_ref(),
                data_type,
                false,
            )
            .map_err(|err| AudioError::Runtime(format!("failed to initialize activation kernel: {err}")))?,
            add: <<Metal as Backend>::Kernels as Kernels>::AudioAddKernel::new(context.as_ref(), data_type)
                .map_err(|err| AudioError::Runtime(format!("failed to initialize add kernel: {err}")))?,
        });
        cache.borrow_mut().insert(key, created.clone());
        Ok(created)
    })
}
