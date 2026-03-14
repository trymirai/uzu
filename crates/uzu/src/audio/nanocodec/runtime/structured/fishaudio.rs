fn fishaudio_dtype_key(data_type: DataType) -> u8 {
    match data_type {
        DataType::F16 => 1,
        DataType::BF16 => 2,
        _ => 0,
    }
}

struct StructuredAudioPostModuleRuntime {
    context: Rc<<Metal as Backend>::Context>,
    model_shape: ModelShape,
    scratch_buffers: ScratchBuffers<Metal>,
    shared_buffers: Rc<RefCell<SharedBuffers<Metal>>>,
    layers: Box<[LayerExecutables<Metal>]>,
    output_norm: RMSNorm<Metal>,
    max_sequence_length: usize,
}

struct StructuredAudioKernelCache {
    half_snake: <<Metal as Backend>::Kernels as Kernels>::AudioHalfSnakeKernel,
    causal_conv1d: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dKernel,
    causal_conv1d_grouped: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedKernel,
    causal_conv1d_grouped_residual: <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dGroupedResidualKernel,
    causal_conv_transpose1d_causal_pad:
        <<Metal as Backend>::Kernels as Kernels>::AudioCausalConvTranspose1dCausalPadKernel,
    conv1d: <<Metal as Backend>::Kernels as Kernels>::AudioConv1dKernel,
    norm_ncs: <<Metal as Backend>::Kernels as Kernels>::AudioNormNcsKernel,
    activation: <<Metal as Backend>::Kernels as Kernels>::ActivationKernel,
    add: <<Metal as Backend>::Kernels as Kernels>::AudioAddKernel,
}

struct FishAudioQuantizerGpuResources {
    data_type: DataType,
    codebook_dim: usize,
    residual_quantizers: usize,
    semantic_cardinality: usize,
    residual_cardinality: usize,
    kernel: <<Metal as Backend>::Kernels as Kernels>::AudioQuantizerDecodeKernel,
    semantic_codebook: Array<Metal>,
    semantic_out_proj: Array<Metal>,
    semantic_out_bias: Array<Metal>,
    residual_codebooks: Array<Metal>,
    residual_out_proj: Array<Metal>,
    residual_out_bias: Array<Metal>,
}

thread_local! {
    static FISHAUDIO_POST_MODULE_RUNTIME_CACHE: RefCell<HashMap<String, Rc<StructuredAudioPostModuleRuntime>>> =
        RefCell::new(HashMap::new());
    static FISHAUDIO_DECODE_CONTEXT_CACHE: RefCell<HashMap<String, Rc<<Metal as Backend>::Context>>> =
        RefCell::new(HashMap::new());
    static FISHAUDIO_KERNEL_CACHE: RefCell<HashMap<usize, Rc<StructuredAudioKernelCache>>> =
        RefCell::new(HashMap::new());
    static FISHAUDIO_VOCODER_GRAPH_CACHE: RefCell<HashMap<usize, Rc<StructuredAudioDecoderGpuGraph>>> =
        RefCell::new(HashMap::new());
    static FISHAUDIO_QUANTIZER_RESOURCES_CACHE: RefCell<HashMap<usize, Rc<FishAudioQuantizerGpuResources>>> =
        RefCell::new(HashMap::new());
}
