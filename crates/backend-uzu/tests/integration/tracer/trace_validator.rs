use std::{
    borrow::Cow,
    cell::RefCell,
    collections::BTreeMap,
    fs::File,
    path::{Path, PathBuf},
    rc::Rc,
};

use backend_uzu::{
    _private::{
        ActivationTrace, AnyModelConfig, AnyTokenCodecConfig, CacheLayers, Classifier, DecoderDecodeInput,
        KVCacheLayer, LanguageModelGeneratorContext, TokenInputs, TransformerLayerConfig,
    },
    array::{Array, ArrayContextExt, ArrayElement, size_for_shape},
    backends::common::{AsBufferRangeRef, Backend, Buffer, Encoder},
    data_type::DataType,
    parameters::{
        HashMetadata, ParameterLoader, ParameterTree, SafeTensorData, read_safetensors_metadata,
        write_safetensors_with_metadata,
    },
    session::{
        config::{DecodingConfig, SpeculatorConfig},
        helpers::{InputProcessor, InputProcessorDefault},
        parameter::{AsyncBatchSize, ConfigResolvableValue, ContextLength, ContextMode, PrefillStepSize, SamplingSeed},
        types::{Error, Input, Message},
    },
};
use num_traits::NumCast;
use serde_json::json;
use tokenizers::Tokenizer;

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
        if !model_path.exists() {
            return Err(Error::ModelFolderNotFound);
        }

        let model_config = load_model_config(model_path)?;
        let context = match model_config {
            AnyModelConfig::ClassifierModelConfig(model_config) => {
                ModelContext::Classifier(Classifier::new(model_path, &model_config)?)
            },
            AnyModelConfig::LanguageModelConfig(model_config) => {
                let prefill_step_size = Self::determine_prefill_step_size(model_path)?;
                let decoding_config = DecodingConfig::new(
                    ContextMode::default(),
                    ContextLength::default(),
                    PrefillStepSize::Custom(prefill_step_size),
                    SpeculatorConfig::default(),
                    SamplingSeed::default(),
                    AsyncBatchSize::default(),
                );
                let mut llm_context = LanguageModelGeneratorContext::new(model_path, &decoding_config, &model_config)?;
                let desired_suffix_length = prefill_step_size.max(decoding_config.generate_suffix_length());
                Self::ensure_llm_context_capacity(&decoding_config, desired_suffix_length, &mut llm_context);
                ModelContext::LanguageModelGenerator(llm_context)
            },
            AnyModelConfig::TTSModelConfig(_) => {
                return Err(Error::InvalidModelConfig("TTS trace export is not supported".to_string()));
            },
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
        let traces_file = File::open(traces_path).map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let (_offset, traces_header) =
            read_safetensors_metadata(&traces_file).map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let token_ids_shape = trace_tensor_shape(&traces_header, "activation_trace.token_ids")?;
        let token_positions_shape = trace_tensor_shape(&traces_header, "activation_trace.token_positions")?;
        let input_metadata = traces_header.metadata.clone().map(|values| values.into_iter().collect());
        let traces_loader = ParameterLoader::new(&traces_file, ctx.context.as_ref())
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let traces_view = traces_loader.tree();
        let (token_ids_array, token_positions_array, token_ids, token_positions) =
            Self::load_trace_inputs(&traces_view, &token_ids_shape, &token_positions_shape)?;
        let traces = Self::run_llm_trace(ctx, &token_ids, &token_positions)?;
        let cache_state_arrays = {
            let cache_layers = ctx.cache_layers.borrow();
            Self::cache_state_arrays(ctx.context.as_ref(), &cache_layers, &traces_header)?
        };

        let mut tensors = vec![
            Self::tensor_from_array("activation_trace.token_ids", &token_ids_array),
            Self::tensor_from_array("activation_trace.token_positions", &token_positions_array),
        ];
        Self::push_activation_trace_tensors(
            &mut tensors,
            &traces,
            &ctx.model_config.decoder_config.transformer_config.layer_configs,
            ExportShape::Batched,
        );
        for (path, array) in &cache_state_arrays {
            tensors.push(Self::tensor_from_array(path, array));
        }
        Self::write_trace_file(output_path, &tensors, input_metadata.as_ref())
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

        ctx.cache_layers.borrow_mut().clear(ctx.context.as_ref());

        let mut encoder = Encoder::<B>::new(ctx.context.as_ref())
            .map_err(|error| Error::UnableToCreateCommandBuffer(error.into()))?;
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
                .map_err(|error| Error::EncodeFailed(Box::new(error)))?;
            if token_ids.len() > 1 {
                cache_layers.commit_short_conv_suffix_states(token_ids.len() - 1, &mut encoder);
            }
        }
        let pending = encoder.end_encoding().submit();
        pending.wait_until_completed().map_err(|error| Error::CommandBufferFailed(Box::new(error)))?;
        Ok(traces)
    }

    fn export_classifier_trace(
        classifier: &mut Classifier<B>,
        traces_path: &Path,
        output_path: &Path,
    ) -> Result<(), Error> {
        let traces_file = File::open(traces_path).map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let (_offset, traces_header) =
            read_safetensors_metadata(&traces_file).map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let token_ids_shape = trace_tensor_shape(&traces_header, "activation_trace.token_ids")?;
        let token_positions_shape = trace_tensor_shape(&traces_header, "activation_trace.token_positions")?;
        let input_metadata = traces_header.metadata.clone().map(|values| values.into_iter().collect());
        let context = classifier.context.context.clone();
        let traces_loader = ParameterLoader::new(&traces_file, context.as_ref())
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let traces_view = traces_loader.tree();
        let (token_ids_array, token_positions_array, token_ids, token_positions) =
            Self::load_trace_inputs(&traces_view, &token_ids_shape, &token_positions_shape)?;
        let (_logits, traces) = classifier.forward_pass_with_traces(&token_ids, &token_positions)?;

        let mut tensors = vec![
            Self::tensor_from_array("activation_trace.token_ids", &token_ids_array),
            Self::tensor_from_array("activation_trace.token_positions", &token_positions_array),
        ];
        Self::push_activation_trace_tensors(
            &mut tensors,
            &traces,
            &classifier.context.model_config.classifier_config.transformer_config.layer_configs,
            ExportShape::Batched,
        );
        Self::write_trace_file(output_path, &tensors, input_metadata.as_ref())
    }

    fn load_trace_inputs(
        traces_view: &ParameterTree<B>,
        token_ids_shape: &[usize],
        token_positions_shape: &[usize],
    ) -> Result<(Array<B>, Array<B>, Vec<u64>, Vec<usize>), Error> {
        let token_ids_array = Self::read_trace_array(traces_view, "activation_trace.token_ids", token_ids_shape)?;
        let token_positions_array =
            Self::read_trace_array(traces_view, "activation_trace.token_positions", token_positions_shape)?;
        Self::validate_trace_input_shape(&token_ids_array, &token_positions_array)?;
        let token_ids = Self::array_as_vec::<i32, u64>(&token_ids_array, "activation_trace.token_ids")?;
        let token_positions =
            Self::array_as_vec::<i32, usize>(&token_positions_array, "activation_trace.token_positions")?;
        Self::validate_token_positions(&token_positions)?;
        Ok((token_ids_array, token_positions_array, token_ids, token_positions))
    }

    fn read_trace_array(
        traces_view: &ParameterTree<B>,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<Array<B>, Error> {
        traces_view
            .leaf(name)
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?
            .validate(expected_shape, DataType::I32)
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?
            .read_array()
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))
    }

    fn validate_trace_input_shape(
        token_ids: &Array<B>,
        token_positions: &Array<B>,
    ) -> Result<(), Error> {
        let &[batch, suffix_length] = token_ids.shape() else {
            return Err(Error::InvalidModelConfig(format!(
                "activation_trace.token_ids must have shape [1, suffix_tokens], got {:?}",
                token_ids.shape()
            )));
        };
        if batch != 1 || suffix_length == 0 {
            return Err(Error::InvalidModelConfig(format!(
                "activation_trace.token_ids must have shape [1, suffix_tokens], got {:?}",
                token_ids.shape()
            )));
        }
        if token_positions.shape() != token_ids.shape() {
            return Err(Error::InvalidModelConfig(format!(
                "activation_trace.token_positions shape {:?} must match activation_trace.token_ids {:?}",
                token_positions.shape(),
                token_ids.shape()
            )));
        }
        Ok(())
    }

    fn validate_token_positions(token_positions: &[usize]) -> Result<(), Error> {
        for (expected, position) in token_positions.iter().copied().enumerate() {
            if position != expected {
                return Err(Error::InvalidModelConfig(format!(
                    "activation_trace.token_positions must be [0, 1, ..., suffix_tokens - 1], got {token_positions:?}"
                )));
            }
        }
        Ok(())
    }

    fn array_as_vec<SourcePrecision: ArrayElement, TargetPrecision: NumCast>(
        array: &Array<B>,
        name: &str,
    ) -> Result<Vec<TargetPrecision>, Error> {
        array
            .as_slice::<SourcePrecision>()
            .iter()
            .map(|value| {
                NumCast::from(*value)
                    .ok_or_else(|| Error::InvalidModelConfig(format!("{name} contains an out-of-range value")))
            })
            .collect()
    }

    fn push_array<'data>(
        tensors: &mut Vec<SafeTensorData<'data>>,
        path: impl Into<String>,
        array: &'data Array<B>,
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

    fn tensor_from_array<'data>(
        name: impl Into<String>,
        array: &'data Array<B>,
    ) -> SafeTensorData<'data> {
        Self::tensor_from_array_with_shape(name, array, array.shape().into())
    }

    fn tensor_from_array_with_shape<'data>(
        name: impl Into<String>,
        array: &'data Array<B>,
        shape: Box<[usize]>,
    ) -> SafeTensorData<'data> {
        SafeTensorData {
            name: name.into(),
            shape,
            data_type: array.data_type(),
            data: Cow::Borrowed(array.as_bytes()),
        }
    }

    fn push_activation_trace_tensors<'data>(
        tensors: &mut Vec<SafeTensorData<'data>>,
        traces: &'data ActivationTrace<B>,
        layer_configs: &[TransformerLayerConfig],
        export_shape: ExportShape,
    ) {
        if let Some(embedding_norm) = &traces.embedding_norm {
            Self::push_array(tensors, "activation_trace.embedding_norm_output", embedding_norm, export_shape);
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
            Self::push_array(tensors, "activation_trace.output_pooling", output_pooling, ExportShape::Native);
        }
        let logits_shape = if traces.output_pooling.is_some() {
            ExportShape::Native
        } else {
            export_shape
        };
        Self::push_array(tensors, "logits", &traces.logits, logits_shape);
    }

    fn cache_state_arrays(
        context: &B::Context,
        cache_layers: &CacheLayers<B>,
        trace_metadata: &HashMetadata,
    ) -> Result<Vec<(String, Array<B>)>, Error> {
        let mut arrays = Vec::new();
        for (index, layer) in cache_layers.iter_layers() {
            if let Some(kv) = layer.as_transformer() {
                if let Some(layer) = kv.as_any().downcast_ref::<KVCacheLayer<B, B::SparseBuffer>>() {
                    Self::push_kv_cache_arrays(&mut arrays, context, trace_metadata, index, layer)?;
                } else if let Some(layer) = kv.as_any().downcast_ref::<KVCacheLayer<B, B::DenseBuffer>>() {
                    Self::push_kv_cache_arrays(&mut arrays, context, trace_metadata, index, layer)?;
                } else {
                    panic!("Unsupported KV cache layer buffer type");
                }
            } else if let Some(ssm) = layer.as_state_space() {
                let path = format!("activation_trace.layer_results.{index}.updated_state.conv_state");
                if let Some(reference_shape) = trace_optional_shape(trace_metadata, &path) {
                    let conv_state = ssm.conv_state.as_ref().ok_or_else(|| {
                        Error::InvalidModelConfig(format!(
                            "{path} is present in the trace but Uzu produced no conv state"
                        ))
                    })?;
                    let array = Self::conv_state_array(
                        context,
                        conv_state,
                        &path,
                        reference_shape,
                        &ssm.conv_shape,
                        ssm.data_type,
                    )?;
                    arrays.push((path, array));
                }
                Self::push_cache_exact_array(
                    &mut arrays,
                    context,
                    trace_metadata,
                    format!("activation_trace.layer_results.{index}.updated_state.ssm_state"),
                    &ssm.ssm_state,
                    &ssm.ssm_shape,
                    ssm.data_type,
                )?;
            } else if let Some(delta) = layer.as_delta_net() {
                Self::push_conv_state_array(
                    &mut arrays,
                    context,
                    trace_metadata,
                    format!("activation_trace.layer_results.{index}.updated_state.conv_state"),
                    &delta.conv_state,
                    &delta.conv_shape,
                    delta.data_type,
                )?;
                Self::push_cache_exact_array(
                    &mut arrays,
                    context,
                    trace_metadata,
                    format!("activation_trace.layer_results.{index}.updated_state.ssm_state"),
                    &delta.ssm_state,
                    &delta.ssm_shape,
                    delta.data_type,
                )?;
            } else if let Some(short_conv) = layer.as_short_conv() {
                Self::push_conv_state_array(
                    &mut arrays,
                    context,
                    trace_metadata,
                    format!("activation_trace.layer_results.{index}.updated_state.conv_state"),
                    &short_conv.conv_state,
                    &short_conv.conv_shape,
                    short_conv.data_type,
                )?;
            } else {
                panic!("Unsupported cache layer type at layer {index}");
            }
        }
        Ok(arrays)
    }

    fn push_kv_cache_arrays<Buf>(
        arrays: &mut Vec<(String, Array<B>)>,
        context: &B::Context,
        trace_metadata: &HashMetadata,
        index: usize,
        layer: &KVCacheLayer<B, Buf>,
    ) -> Result<(), Error>
    where
        Buf: Buffer<Backend = B>,
    {
        Self::push_cache_prefix_array(
            arrays,
            context,
            trace_metadata,
            format!("activation_trace.layer_results.{index}.updated_state.keys"),
            &layer.keys,
            &layer.shape,
            layer.data_type,
        )?;
        Self::push_cache_prefix_array(
            arrays,
            context,
            trace_metadata,
            format!("activation_trace.layer_results.{index}.updated_state.values"),
            &layer.values,
            &layer.shape,
            layer.data_type,
        )
    }

    fn push_cache_prefix_array<Source>(
        arrays: &mut Vec<(String, Array<B>)>,
        context: &B::Context,
        trace_metadata: &HashMetadata,
        path: String,
        source: &Source,
        native_shape: &[usize],
        data_type: DataType,
    ) -> Result<(), Error>
    where
        Source: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
    {
        let Some(reference_shape) = trace_optional_shape(trace_metadata, &path) else {
            return Ok(());
        };
        let array = Self::cache_prefix_array(context, source, &path, reference_shape, native_shape, data_type)?;
        arrays.push((path, array));
        Ok(())
    }

    fn push_cache_exact_array<Source>(
        arrays: &mut Vec<(String, Array<B>)>,
        context: &B::Context,
        trace_metadata: &HashMetadata,
        path: String,
        source: &Source,
        native_shape: &[usize],
        data_type: DataType,
    ) -> Result<(), Error>
    where
        Source: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
    {
        let Some(reference_shape) = trace_optional_shape(trace_metadata, &path) else {
            return Ok(());
        };
        let array = Self::cache_exact_array(context, source, &path, reference_shape, native_shape, data_type)?;
        arrays.push((path, array));
        Ok(())
    }

    fn push_conv_state_array<Source>(
        arrays: &mut Vec<(String, Array<B>)>,
        context: &B::Context,
        trace_metadata: &HashMetadata,
        path: String,
        source: &Source,
        native_shape: &[usize; 2],
        data_type: DataType,
    ) -> Result<(), Error>
    where
        Source: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
    {
        let Some(reference_shape) = trace_optional_shape(trace_metadata, &path) else {
            return Ok(());
        };
        let array = Self::conv_state_array(context, source, &path, reference_shape, native_shape, data_type)?;
        arrays.push((path, array));
        Ok(())
    }

    fn cache_prefix_array<Source>(
        context: &B::Context,
        source: &Source,
        path: &str,
        reference_shape: &[usize],
        native_shape: &[usize],
        data_type: DataType,
    ) -> Result<Array<B>, Error>
    where
        Source: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
    {
        let aligned = match reference_shape {
            shape if shape == native_shape => true,
            [tokens, rest @ ..] if reference_shape.len() == native_shape.len() => {
                *tokens <= native_shape[0] && rest == &native_shape[1..]
            },
            [1, tokens, rest @ ..] if reference_shape.len() == native_shape.len() + 1 => {
                *tokens <= native_shape[0] && rest == &native_shape[1..]
            },
            _ => false,
        };
        if !aligned {
            return Err(Error::InvalidModelConfig(format!(
                "{path} trace shape {reference_shape:?} is incompatible with Uzu cache shape {native_shape:?}"
            )));
        }
        Self::copy_buffer_prefix_to_array(context, source, path, reference_shape, data_type)
    }

    fn cache_exact_array<Source>(
        context: &B::Context,
        source: &Source,
        path: &str,
        reference_shape: &[usize],
        native_shape: &[usize],
        data_type: DataType,
    ) -> Result<Array<B>, Error>
    where
        Source: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
    {
        let aligned =
            reference_shape == native_shape || matches!(reference_shape, [1, rest @ ..] if rest == native_shape);
        if !aligned {
            return Err(Error::InvalidModelConfig(format!(
                "{path} trace shape {reference_shape:?} is incompatible with Uzu cache shape {native_shape:?}"
            )));
        }
        Self::copy_buffer_prefix_to_array(context, source, path, reference_shape, data_type)
    }

    fn conv_state_array<Source>(
        context: &B::Context,
        source: &Source,
        path: &str,
        reference_shape: &[usize],
        native_shape: &[usize; 2],
        data_type: DataType,
    ) -> Result<Array<B>, Error>
    where
        Source: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
    {
        let [source_dim, source_history] = *native_shape;
        let (target_time, dim) = match reference_shape {
            [time, dim] => (*time, *dim),
            [1, time, dim] => (*time, *dim),
            _ => {
                return Err(Error::InvalidModelConfig(format!(
                    "{path} trace shape {reference_shape:?} is incompatible with Uzu conv state shape {native_shape:?}"
                )));
            },
        };
        if dim != source_dim || target_time != source_history {
            return Err(Error::InvalidModelConfig(format!(
                "{path} trace shape {reference_shape:?} is incompatible with Uzu conv state shape {native_shape:?}"
            )));
        }

        let source_array = Self::copy_buffer_prefix_to_array(context, source, path, native_shape, data_type)?;
        let mut target_array = context.create_array_uninitialized(reference_shape, data_type);
        let element_bytes = data_type.size_in_bytes();
        for dim_index in 0..dim {
            for history_index in 0..source_history {
                let source_offset = (dim_index * source_history + history_index) * element_bytes;
                let target_offset = (history_index * dim + dim_index) * element_bytes;
                target_array.as_bytes_mut()[target_offset..target_offset + element_bytes]
                    .copy_from_slice(&source_array.as_bytes()[source_offset..source_offset + element_bytes]);
            }
        }
        Ok(target_array)
    }

    fn copy_buffer_prefix_to_array<Source>(
        context: &B::Context,
        source: &Source,
        path: &str,
        shape: &[usize],
        data_type: DataType,
    ) -> Result<Array<B>, Error>
    where
        Source: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
    {
        let target_size = size_for_shape(shape, data_type);
        let source_size = source.as_buffer_range_ref().range().len();
        if target_size > source_size {
            return Err(Error::InvalidModelConfig(format!(
                "{path} trace shape {shape:?} requires {target_size} bytes but Uzu cache has {source_size}"
            )));
        }
        let mut array = context.create_array_uninitialized(shape, data_type);
        let mut encoder =
            Encoder::<B>::new(context).map_err(|error| Error::UnableToCreateCommandBuffer(error.into()))?;
        encoder.encode_copy(source, 0..target_size, array.allocation_mut(), 0..target_size);
        encoder
            .end_encoding()
            .submit()
            .wait_until_completed()
            .map_err(|error| Error::CommandBufferFailed(Box::new(error)))?;
        Ok(array)
    }

    fn write_trace_file(
        output_path: &Path,
        tensors: &[SafeTensorData<'_>],
        metadata: Option<&BTreeMap<String, String>>,
    ) -> Result<(), Error> {
        let mut file = File::create_new(output_path).map_err(|error| Error::UnableToWriteTrace(Box::new(error)))?;
        write_safetensors_with_metadata(&mut file, tensors, metadata)
            .map_err(|error| Error::UnableToWriteTrace(Box::new(error)))
    }

    fn determine_prefill_step_size(model_path: &Path) -> Result<usize, Error> {
        let traces_path = model_path.join("traces.safetensors");
        let file = File::open(&traces_path).map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let (_header_len, metadata) =
            read_safetensors_metadata(&file).map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let token_shape =
            metadata.tensors.get("activation_trace.token_ids").map(|tensor| tensor.shape.as_slice()).ok_or_else(
                || Error::InvalidModelConfig("missing required trace tensor activation_trace.token_ids".to_string()),
            )?;
        let &[batch, suffix_length] = token_shape else {
            return Err(Error::InvalidModelConfig(format!(
                "activation_trace.token_ids must have shape [1, suffix_tokens], got {token_shape:?}"
            )));
        };
        if batch != 1 || suffix_length == 0 {
            return Err(Error::InvalidModelConfig(format!(
                "activation_trace.token_ids must have shape [1, suffix_tokens], got {token_shape:?}"
            )));
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
    }
}

pub fn export_tokenization_trace(
    model_path: &Path,
    output_path: &Path,
    message: &str,
) -> Result<(), Error> {
    let model_config = load_model_config(model_path)?;
    let AnyModelConfig::LanguageModelConfig(model_config) = model_config else {
        return Err(Error::InvalidModelConfig("tokenization trace export requires a language model".to_string()));
    };
    let AnyTokenCodecConfig::ChatCodecConfig(token_codec_config) = model_config.token_codec_config else {
        return Err(Error::InvalidModelConfig("tokenization trace export requires a chat token codec".to_string()));
    };
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(Error::UnableToLoadTokenizer)?;
    let messages = tokenization_trace_messages(token_codec_config.default_system_prompt.clone(), message);
    let input = Input::Messages(messages.clone());
    let add_generation_prompt = true;
    let enable_thinking = true;
    let rendered_request = InputProcessorDefault::new(token_codec_config.clone()).process(
        &input,
        add_generation_prompt,
        enable_thinking,
    )?;
    let encoding = tokenizer.encode(rendered_request.as_str(), false).map_err(Error::UnableToEncodeText)?;
    let token_ids = encoding
        .get_ids()
        .iter()
        .map(|token| i32::try_from(*token).map_err(|_| Error::InvalidModelConfig("token id exceeds int32".to_string())))
        .collect::<Result<Vec<_>, _>>()?;
    if token_ids.is_empty() {
        return Err(Error::InvalidModelConfig("tokenization trace must contain at least one token".to_string()));
    }
    let token_positions = (0..token_ids.len())
        .map(|position| {
            i32::try_from(position).map_err(|_| Error::InvalidModelConfig("token position exceeds int32".to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let shape = [1, token_ids.len()];
    let tensors = [
        tensor_from_i32_slice("activation_trace.token_ids", &shape, &token_ids),
        tensor_from_i32_slice("activation_trace.token_positions", &shape, &token_positions),
    ];
    let request = json!({
        "add_generation_prompt": add_generation_prompt,
        "messages": messages.iter().map(|message| message.resolve(&token_codec_config)).collect::<Vec<_>>(),
        "bos_token": token_codec_config.bos_token,
        "eos_token": token_codec_config.eos_token,
        "enable_thinking": enable_thinking,
    });
    let mut trace_metadata = BTreeMap::new();
    trace_metadata.insert("add_special_tokens".to_string(), "false".to_string());
    trace_metadata.insert("prompt_template".to_string(), token_codec_config.prompt_template);
    trace_metadata.insert("rendered_request".to_string(), rendered_request);
    trace_metadata.insert(
        "request".to_string(),
        serde_json::to_string(&request).map_err(|error| Error::UnableToWriteTrace(Box::new(error)))?,
    );
    trace_metadata.insert(
        "tokens".to_string(),
        serde_json::to_string(encoding.get_tokens()).map_err(|error| Error::UnableToWriteTrace(Box::new(error)))?,
    );

    let mut file = File::create_new(output_path).map_err(|error| Error::UnableToWriteTrace(Box::new(error)))?;
    write_safetensors_with_metadata(&mut file, &tensors, Some(&trace_metadata))
        .map_err(|error| Error::UnableToWriteTrace(Box::new(error)))
}

fn tokenization_trace_messages(
    default_system_prompt: Option<String>,
    message: &str,
) -> Vec<Message> {
    let mut messages = vec![Message::user(message.to_string())];
    if let Some(default_system_prompt) = default_system_prompt {
        messages.insert(0, Message::system(default_system_prompt));
    }
    messages
}

fn load_model_config(model_path: &Path) -> Result<AnyModelConfig, Error> {
    let config_file = File::open(model_path.join("config.json"))?;
    serde_json::from_reader(std::io::BufReader::new(config_file)).map_err(Error::UnableToDeserializeConfig)
}

fn trace_tensor_shape(
    metadata: &backend_uzu::parameters::HashMetadata,
    name: &str,
) -> Result<Vec<usize>, Error> {
    metadata
        .tensors
        .get(name)
        .map(|tensor| tensor.shape.clone())
        .ok_or_else(|| Error::InvalidModelConfig(format!("missing required trace tensor {name}")))
}

fn trace_optional_shape<'metadata>(
    metadata: &'metadata HashMetadata,
    name: &str,
) -> Option<&'metadata [usize]> {
    metadata.tensors.get(name).map(|tensor| tensor.shape.as_slice())
}

fn tensor_from_i32_slice(
    name: impl Into<String>,
    shape: &[usize; 2],
    values: &[i32],
) -> SafeTensorData<'static> {
    SafeTensorData {
        name: name.into(),
        shape: Box::new(*shape),
        data_type: DataType::I32,
        data: Cow::Owned(values.iter().flat_map(|value| value.to_le_bytes()).collect()),
    }
}

mod tests {
    use backend_uzu::{
        array::ArrayContextExt,
        backends::{
            common::{Backend, Context},
            cpu::Cpu,
        },
        data_type::DataType,
        session::types::{Error, Role},
    };

    use super::{TraceValidator, tokenization_trace_messages};

    #[test]
    fn test_tokenization_trace_messages_preserve_default_system_prompt() {
        let messages = tokenization_trace_messages(Some("Stay brief".to_string()), "Hello");

        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content, "Stay brief");
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[1].content, "Hello");
    }

    #[test]
    fn test_conv_state_array_matches_lalamo_state_layout() {
        let context = <Cpu as Backend>::Context::new().expect("create CPU context");
        let source = context.create_array_from(&[2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let result = TraceValidator::<Cpu>::conv_state_array(
            context.as_ref(),
            source.allocation(),
            "activation_trace.layer_results.0.updated_state.conv_state",
            &[1, 3, 2],
            &[2, 3],
            DataType::F32,
        )
        .expect("export conv state");

        assert_eq!(result.shape(), &[1, 3, 2]);
        assert_eq!(result.as_slice::<f32>(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_conv_state_array_transposes_square_state() {
        let context = <Cpu as Backend>::Context::new().expect("create CPU context");
        let source = context.create_array_from(&[2, 2], &[1.0f32, 2.0, 3.0, 4.0]);

        let result = TraceValidator::<Cpu>::conv_state_array(
            context.as_ref(),
            source.allocation(),
            "activation_trace.layer_results.0.updated_state.conv_state",
            &[1, 2, 2],
            &[2, 2],
            DataType::F32,
        )
        .expect("export conv state");

        assert_eq!(result.shape(), &[1, 2, 2]);
        assert_eq!(result.as_slice::<f32>(), &[1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_conv_state_array_rejects_history_padding() {
        let context = <Cpu as Backend>::Context::new().expect("create CPU context");
        let source = context.create_array_from(&[2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let result = TraceValidator::<Cpu>::conv_state_array(
            context.as_ref(),
            source.allocation(),
            "activation_trace.layer_results.0.updated_state.conv_state",
            &[1, 5, 2],
            &[2, 3],
            DataType::F32,
        );

        assert!(matches!(result, Err(Error::InvalidModelConfig(message)) if message.contains("incompatible")));
    }

    #[test]
    fn test_validate_token_positions_requires_sequential_positions() {
        assert!(TraceValidator::<Cpu>::validate_token_positions(&[0, 1, 2]).is_ok());

        let result = TraceValidator::<Cpu>::validate_token_positions(&[0, 2]);

        assert!(matches!(result, Err(Error::InvalidModelConfig(message)) if message.contains("token_positions")));
    }
}
