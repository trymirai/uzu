use std::{
    cell::RefCell,
    fs::File,
    path::{Path, PathBuf},
    rc::Rc,
};

use backend_uzu::{
    _private::{
        ActivationTrace, CacheLayers, Classifier, DecoderDecodeInput, LanguageModelGeneratorContext, ModelConfig,
        ModelMetadata, ModelType, ParameterTree, Sampling, TokenInputs, TransformerLayerConfig,
    },
    Array, ArrayElement, DataType, ParameterLoader, SafeTensorData,
    backends::common::{Backend, Encoder, allocation_as_bytes, kernel::kv_cache_update::KVCacheUpdate},
    read_safetensors_metadata,
    session::{
        config::{DecodingConfig, SpeculatorConfig},
        parameter::{AsyncBatchSize, ConfigResolvableValue, ContextLength, ContextMode, PrefillStepSize, SamplingSeed},
        types::Error,
    },
    write_safetensors,
};
use num_traits::NumCast;

#[derive(Clone, Copy)]
enum ExportShape {
    Native,
    Batched,
}

enum ModelContext<B: Backend> {
    LanguageModelGenerator(LanguageModelGeneratorContext<B>),
    Classifier(Classifier<B>),
}

pub struct TraceValidator<B: Backend> {
    model_path: PathBuf,
    context: ModelContext<B>,
}

impl<B: Backend> TraceValidator<B> {
    pub fn new(model_path: &Path) -> Result<Self, Error> {
        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(Error::ModelFolderNotFound);
        }

        let config_file = File::open(&config_path).map_err(|_| Error::UnableToLoadConfig)?;
        let raw_metadata: ModelMetadata<ModelConfig> =
            serde_json::from_reader(std::io::BufReader::new(config_file)).map_err(|_| Error::UnableToLoadConfig)?;

        let context = match raw_metadata.model_type.clone() {
            ModelType::ClassifierModel => {
                let ModelConfig::ClassifierModel(model_config) = raw_metadata.model_config.clone() else {
                    return Err(Error::UnableToLoadConfig);
                };
                let metadata = ModelMetadata {
                    toolchain_version: raw_metadata.toolchain_version,
                    vendor: raw_metadata.vendor,
                    family: raw_metadata.family,
                    name: raw_metadata.name,
                    size: raw_metadata.size,
                    quantization: raw_metadata.quantization,
                    repo: raw_metadata.repo,
                    use_cases: raw_metadata.use_cases,
                    model_type: raw_metadata.model_type,
                    model_config,
                    grammar_start_tokens: raw_metadata.grammar_start_tokens,
                };
                ModelContext::Classifier(Classifier::new(model_path, &metadata)?)
            },
            ModelType::LanguageModel => {
                let ModelConfig::LanguageModel(model_config) = raw_metadata.model_config.clone() else {
                    return Err(Error::UnableToLoadConfig);
                };
                let metadata = ModelMetadata {
                    toolchain_version: raw_metadata.toolchain_version,
                    vendor: raw_metadata.vendor,
                    family: raw_metadata.family,
                    name: raw_metadata.name,
                    size: raw_metadata.size,
                    quantization: raw_metadata.quantization,
                    repo: raw_metadata.repo,
                    use_cases: raw_metadata.use_cases,
                    model_type: raw_metadata.model_type,
                    model_config,
                    grammar_start_tokens: raw_metadata.grammar_start_tokens,
                };
                let prefill_step_size = Self::determine_prefill_step_size(model_path)?;
                let decoding_config = DecodingConfig::new(
                    ContextMode::default(),
                    ContextLength::default(),
                    PrefillStepSize::Custom(prefill_step_size),
                    SpeculatorConfig::default(),
                    SamplingSeed::default(),
                    AsyncBatchSize::default(),
                );
                let mut llm_context = LanguageModelGeneratorContext::new(model_path, &decoding_config, &metadata)?;
                let desired_suffix_length = prefill_step_size.max(decoding_config.generate_suffix_length());
                Self::ensure_llm_context_capacity(&decoding_config, desired_suffix_length, &mut llm_context);
                ModelContext::LanguageModelGenerator(llm_context)
            },
            ModelType::TtsModel => return Err(Error::UnableToLoadConfig),
        };

        Ok(Self {
            model_path: model_path.to_path_buf(),
            context,
        })
    }

    pub fn export_trace(
        &mut self,
        output_path: &Path,
    ) -> Result<(), Error> {
        let traces_path = self.model_path.join("traces.safetensors");
        if !traces_path.exists() {
            return Err(Error::UnableToLoadWeights);
        }

        match &mut self.context {
            ModelContext::LanguageModelGenerator(ctx) => Self::export_llm_trace(ctx, &traces_path, output_path),
            ModelContext::Classifier(classifier) => {
                Self::export_classifier_trace(classifier, &traces_path, output_path)
            },
        }
    }

    fn export_llm_trace(
        ctx: &LanguageModelGeneratorContext<B>,
        traces_path: &Path,
        output_path: &Path,
    ) -> Result<(), Error> {
        let traces_file = File::open(traces_path).map_err(|_| Error::UnableToLoadWeights)?;
        let traces_loader =
            ParameterLoader::new(&traces_file, ctx.context.as_ref()).map_err(|_| Error::UnableToLoadWeights)?;
        let traces_view = traces_loader.tree();
        let (token_ids_array, token_positions_array, token_ids, token_positions) =
            Self::load_trace_inputs(&traces_view)?;
        let traces = Self::run_llm_trace(ctx, &token_ids, &token_positions)?;

        let mut tensors = vec![
            Self::tensor_from_array("activation_trace.token_ids", &token_ids_array),
            Self::tensor_from_array("activation_trace.token_positions", &token_positions_array),
        ];
        Self::push_activation_trace_tensors(
            &mut tensors,
            &traces,
            &ctx.model_config.model_config.transformer_config.layer_configs,
            ExportShape::Batched,
        );
        Self::write_trace_file(output_path, &tensors)
    }

    fn run_llm_trace(
        ctx: &LanguageModelGeneratorContext<B>,
        token_ids: &[u64],
        token_positions: &[usize],
    ) -> Result<ActivationTrace<B>, Error> {
        let token_inputs = TokenInputs::new_llm(
            ctx.context.as_ref(),
            &ctx.model_shape,
            token_ids,
            None,
            token_positions,
            None,
            None,
            /*sampling_start=*/ 0,
            /*sampling_length=*/ token_ids.len(),
        );
        let mut traces = ActivationTrace::new_llm(ctx.context.as_ref(), &ctx.model_shape, token_ids.len());

        let mut encoder =
            Encoder::<B>::new(ctx.context.as_ref()).map_err(|err| Error::UnableToCreateCommandBuffer(err.into()))?;
        {
            let mut cache_layers = ctx.cache_layers.borrow_mut();
            cache_layers.prepare_for_forward_pass(ctx.context.as_ref(), token_ids.len());
            let decoder_arguments = token_inputs.decoder_arguments(
                ctx.shared_buffers.as_ref(),
                Some(&mut *cache_layers),
                token_ids.len(),
                /*sampling_start=*/ 0,
                /*sampling_length=*/ token_ids.len(),
                #[cfg(feature = "tracing")]
                Some(&mut traces),
            );
            ctx.executables
                .encode_decode(
                    decoder_arguments,
                    DecoderDecodeInput::TokenIds(token_inputs.token_ids()),
                    None,
                    &mut encoder,
                )
                .map_err(|err| Error::EncodeFailed(Box::new(err)))?;
        }
        let pending = encoder.end_encoding().submit();
        pending.wait_until_completed().map_err(|err| Error::CommandBufferFailed(Box::new(err)))?;
        Ok(traces)
    }

    fn export_classifier_trace(
        classifier: &mut Classifier<B>,
        traces_path: &Path,
        output_path: &Path,
    ) -> Result<(), Error> {
        let traces_file = File::open(traces_path).map_err(|_| Error::UnableToLoadWeights)?;
        let context = classifier.context.context.clone();
        let traces_loader =
            ParameterLoader::new(&traces_file, context.as_ref()).map_err(|_| Error::UnableToLoadWeights)?;
        let traces_view = traces_loader.tree();
        let (token_ids_array, token_positions_array, token_ids, token_positions) =
            Self::load_trace_inputs(&traces_view)?;
        let (_logits, traces) =
            classifier.forward_pass_with_traces(&token_ids, &token_positions).map_err(|_| Error::GenerateFailed)?;

        let mut tensors = vec![
            Self::tensor_from_array("activation_trace.token_ids", &token_ids_array),
            Self::tensor_from_array("activation_trace.token_positions", &token_positions_array),
        ];
        Self::push_activation_trace_tensors(
            &mut tensors,
            &traces,
            &classifier.context.model_config.model_config.transformer_config.layer_configs,
            ExportShape::Native,
        );
        Self::write_trace_file(output_path, &tensors)
    }

    fn load_trace_inputs(
        traces_view: &ParameterTree<B::Context>
    ) -> Result<(Array<B>, Array<B>, Vec<u64>, Vec<usize>), Error> {
        let token_ids_array =
            traces_view.leaf_array("activation_trace.token_ids").map_err(|_| Error::UnableToLoadWeights)?;
        let token_positions_array =
            traces_view.leaf_array("activation_trace.token_positions").map_err(|_| Error::UnableToLoadWeights)?;
        Self::validate_trace_input_shape(&token_ids_array, &token_positions_array)?;
        let token_ids = Self::array_as_vec::<i32, u64>(&token_ids_array)?;
        let token_positions = Self::array_as_vec::<i32, usize>(&token_positions_array)?;
        Ok((token_ids_array, token_positions_array, token_ids, token_positions))
    }

    fn validate_trace_input_shape(
        token_ids: &Array<B>,
        token_positions: &Array<B>,
    ) -> Result<(), Error> {
        let &[batch, suffix_length] = token_ids.shape() else {
            return Err(Error::UnableToLoadWeights);
        };
        if batch != 1 || suffix_length == 0 || token_positions.shape() != token_ids.shape() {
            return Err(Error::UnableToLoadWeights);
        }
        Ok(())
    }

    fn array_as_vec<SourcePrecision: ArrayElement, TargetPrecision: NumCast>(
        array: &Array<B>
    ) -> Result<Vec<TargetPrecision>, Error> {
        let slice = array.as_slice::<SourcePrecision>();
        slice.iter().map(|value| NumCast::from(*value).ok_or(Error::UnableToLoadWeights)).collect()
    }

    fn push_array(
        tensors: &mut Vec<SafeTensorData>,
        path: impl Into<String>,
        array: &Array<B>,
        export_shape: ExportShape,
    ) {
        let path = path.into();
        let tensor = match export_shape {
            ExportShape::Native => Self::tensor_from_array(path, array),
            ExportShape::Batched => {
                let shape = std::iter::once(1).chain(array.shape().iter().copied()).collect::<Vec<_>>();
                Self::tensor_from_array_with_shape(path, array, shape.into_boxed_slice())
            },
        };
        tensors.push(tensor);
    }

    fn tensor_from_array(
        name: impl Into<String>,
        array: &Array<B>,
    ) -> SafeTensorData {
        Self::tensor_from_array_with_shape(name, array, array.shape().into())
    }

    fn tensor_from_array_with_shape(
        name: impl Into<String>,
        array: &Array<B>,
        shape: Box<[usize]>,
    ) -> SafeTensorData {
        SafeTensorData {
            name: name.into(),
            shape,
            data_type: array.data_type(),
            data: allocation_as_bytes(array.allocation()).into(),
        }
    }

    fn push_activation_trace_tensors(
        tensors: &mut Vec<SafeTensorData>,
        traces: &ActivationTrace<B>,
        layer_configs: &[TransformerLayerConfig],
        export_shape: ExportShape,
    ) {
        if let Some(embedding_norm) = &traces.embedding_norm {
            Self::push_array(tensors, "activation_trace.embedding_norm", embedding_norm, export_shape);
        }

        for (index, layer_traces) in traces.layer_results.iter().enumerate() {
            let layer_config = &layer_configs[index];
            let path = |suffix: &str| format!("activation_trace.layer_results.{index}.activation_trace.{suffix}");

            Self::push_array(tensors, path("inputs"), &layer_traces.inputs, export_shape);
            Self::push_array(tensors, path("pre_mixer_norm"), &layer_traces.pre_attention_norm, export_shape);
            Self::push_array(tensors, path("mixer"), &layer_traces.attention, export_shape);
            if layer_config.post_mixer_norm_config.is_some() {
                Self::push_array(tensors, path("post_mixer_norm"), &layer_traces.post_attention_norm, export_shape);
            }
            Self::push_array(tensors, path("mlp_inputs"), &layer_traces.mlp_inputs, export_shape);
            Self::push_array(tensors, path("pre_mlp_norm"), &layer_traces.pre_mlp_norm, export_shape);
            Self::push_array(tensors, path("mlp"), &layer_traces.mlp, export_shape);
            if layer_config.post_mlp_norm_config.is_some() {
                Self::push_array(tensors, path("post_mlp_norm"), &layer_traces.post_mlp_norm, export_shape);
            }
            Self::push_array(
                tensors,
                format!("activation_trace.layer_results.{index}.outputs"),
                &layer_traces.outputs,
                export_shape,
            );
        }

        Self::push_array(tensors, "activation_trace.output_norm", &traces.output_norm, export_shape);
        if let Some(output_pooling) = &traces.output_pooling {
            Self::push_array(tensors, "activation_trace.output_pooling", output_pooling, export_shape);
        }
        Self::push_array(tensors, "logits", &traces.logits, export_shape);
    }

    fn write_trace_file(
        output_path: &Path,
        tensors: &[SafeTensorData],
    ) -> Result<(), Error> {
        let mut file = File::create_new(output_path).map_err(|_| Error::UnableToWriteTrace)?;
        write_safetensors(&mut file, tensors).map_err(|_| Error::UnableToWriteTrace)
    }

    fn determine_prefill_step_size(model_path: &Path) -> Result<usize, Error> {
        let traces_path = model_path.join("traces.safetensors");
        let file = File::open(&traces_path).map_err(|_| Error::UnableToLoadWeights)?;
        let (_header_len, metadata) = read_safetensors_metadata(&file).map_err(|_| Error::UnableToLoadWeights)?;
        let tensor = metadata.tensors.get("activation_trace.token_ids").ok_or(Error::UnableToLoadWeights)?;
        let &[batch, suffix_length] = tensor.shape.as_slice() else {
            return Err(Error::UnableToLoadWeights);
        };
        if batch != 1 || suffix_length == 0 {
            return Err(Error::UnableToLoadWeights);
        }
        Ok(suffix_length)
    }

    fn ensure_llm_context_capacity(
        decoding_config: &DecodingConfig,
        desired_suffix_length: usize,
        context: &mut LanguageModelGeneratorContext<B>,
    ) {
        let resolved_prefix_length = context.get_context_length(decoding_config);
        let current_suffix_length = std::cmp::max(
            decoding_config.prefill_step_size.resolve(&context.model_config),
            decoding_config.generate_suffix_length(),
        );

        if desired_suffix_length <= current_suffix_length {
            return;
        }

        context.cache_layers = Rc::new(RefCell::new(CacheLayers::new(
            context.context.as_ref(),
            &context.model_shape,
            resolved_prefix_length,
            desired_suffix_length,
        )));

        let intermediate_dtype: DataType =
            context.model_config.model_config.transformer_config.output_norm_config.scale_precision.into();

        context.kv_cache_update = Box::new(
            KVCacheUpdate::new(context.context.as_ref(), intermediate_dtype, resolved_prefix_length)
                .expect("Failed to create KV cache update kernel"),
        );

        context.gpu_sampler =
            Sampling::new(context.context.as_ref(), intermediate_dtype).expect("Failed to create sampling kernel");
    }
}
