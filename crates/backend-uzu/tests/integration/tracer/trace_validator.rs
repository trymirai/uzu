//! Unified trace validation for any model type (LLM or classifier).
//!
//! This module provides a single `TraceValidator` that can validate activation
//! traces for any model type. It automatically detects the model type from the
//! config and runs the appropriate validation.

use std::{
    cell::RefCell,
    collections::HashMap,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    rc::Rc,
};

use backend_uzu::{
    _private::{
        ActivationTrace, AnyModelConfig, CacheLayers, Classifier, DecoderDecodeInput, KVCacheLayer, KVCacheLayerTrait,
        LanguageModelGeneratorContext, TokenInputs,
    },
    array::{Array, ArrayElement},
    backends::common::{Allocation, AllocationType, Backend, Context, Encoder},
    data_type::DataType,
    parameters::{ParameterLoader, ParameterLoaderError, ParameterTree, read_safetensors_metadata},
    session::{
        config::{DecodingConfig, SpeculatorConfig},
        parameter::{AsyncBatchSize, ConfigResolvableValue, ContextLength, ContextMode, PrefillStepSize, SamplingSeed},
        types::Error,
    },
};
use half::{bf16, f16};
use ndarray::{ArrayView, IxDyn, s};
use num_traits::NumCast;

use crate::common;

fn argmax<T: ArrayElement>(input: &[T]) -> usize {
    input
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(best_index, best_value), (index, &value)| {
            let value_f32 = NumCast::from(value).unwrap_or(f32::NEG_INFINITY);
            if value_f32 > best_value || (value_f32 == best_value && index < best_index) {
                (index, value_f32)
            } else {
                (best_index, best_value)
            }
        })
        .0
}

fn sample_argmax<T: ArrayElement>(logits: ArrayView<T, IxDyn>) -> Vec<u64> {
    let mut result = Vec::with_capacity(logits.shape()[0]);
    for row in logits.rows() {
        let max_index = if let Some(slice) = row.as_slice_memory_order() {
            argmax(slice)
        } else {
            let data: Vec<T> = row.iter().copied().collect();
            argmax(&data)
        };
        result.push(max_index as u64);
    }
    result
}
// ============================================================================
// Validation Types
// ============================================================================

/// Metrics from validating a single tensor.
pub struct TracerValidationMetrics {
    pub atol: f32,
    pub rtol: f32,
    pub rms_rtol: f32,
    #[allow(unused)]
    pub fraction_of_allowed_violations: f32,
    pub reference_shape: Vec<usize>,
    pub result_shape: Vec<usize>,
    pub num_violations: usize,
    pub max_allowed_violations: usize,
    pub max_err_idx: usize,
    pub max_err: f32,
    pub max_err_rel: f32,
    pub max_err_reference_value: f32,
    pub rms_diff: f32,
    pub rms_result: f32,
    pub rms_reference: f32,
    pub rel_rms_reference: f32,
    pub diff_max: f32,
    pub diff_avg: f32,
    pub result_nan: bool,
}

impl TracerValidationMetrics {
    pub fn is_valid(&self) -> bool {
        (self.num_violations <= self.max_allowed_violations || self.rel_rms_reference <= self.rms_rtol)
            && !self.result_nan
    }

    pub fn message(&self) -> String {
        if self.result_nan {
            return "Result contains NaN values".to_string();
        }

        let allowed_violation_explainer = if self.max_allowed_violations > 0 {
            format!(" (Max {} allowed)", self.max_allowed_violations)
        } else {
            String::new()
        };

        let reference_size: usize = self.reference_shape.iter().product();

        format!(
            "{} violations > {:.1e} + {:.2}% out of total {} elements{}.\n\
            Worst violation: {:.3} ({:.2}%) at index {} (reference value: {:.3}).\n\
            Error RMS: {:.3}.\n\
            RMS of result: {:.3}, RMS of reference: {:.3}.\n\
            Relative error RMS: {:.2}% of RMS of reference (Max {:.2}% allowed).\n\
            Shape: {:?}\n\
            Max diff: {:.3}, Avg diff: {:.3}",
            self.num_violations,
            self.atol,
            self.rtol * 100.0,
            reference_size,
            allowed_violation_explainer,
            self.max_err,
            self.max_err_rel * 100.0,
            self.max_err_idx,
            self.max_err_reference_value,
            self.rms_diff,
            self.rms_result,
            self.rms_reference,
            self.rel_rms_reference * 100.0,
            self.rms_rtol * 100.0,
            self.result_shape,
            self.diff_max,
            self.diff_avg,
        )
    }
}

/// Result of validating a single tensor.
pub struct TracerValidationResult {
    pub name: String,
    pub metrics: TracerValidationMetrics,
}

/// Results from validating all traces.
pub struct TracerValidationResults {
    pub suffix_length: usize,
    pub results: Vec<TracerValidationResult>,
    pub tokens_violation_indices: Vec<usize>,
}

impl TracerValidationResults {
    pub fn number_of_tokens_violations(&self) -> usize {
        self.tokens_violation_indices.len()
    }

    pub fn number_of_allowed_tokens_violations(&self) -> usize {
        let threshold: f64 = 0.01;
        (self.suffix_length as f64 * threshold).ceil() as usize
    }
}

/// Transform to apply to produced arrays before comparison.
pub enum ArrayTransform {
    /// Slice KV cache to match expected shape.
    KVCacheSlice {
        start: usize,
    },
    /// Gather rows by traced token positions.
    PositionSlice(Vec<usize>),
    /// Transform SSM conv state layout.
    SsmConvState,
}

// ============================================================================
// Model Context (internal)
// ============================================================================

#[derive(Clone, Copy)]
struct LayerTraceRequirements {
    post_mixer_norm: bool,
    post_mlp_norm: bool,
}

enum ModelContext<B: Backend> {
    LanguageModelGenerator {
        context: LanguageModelGeneratorContext<B>,
        layer_trace_requirements: Box<[LayerTraceRequirements]>,
    },
    Classifier {
        classifier: Classifier<B>,
        layer_trace_requirements: Box<[LayerTraceRequirements]>,
    },
}

// ============================================================================
// Unified TraceValidator
// ============================================================================

/// Unified trace validator for any model type.
///
/// Automatically detects whether the model is an LLM or classifier and
/// runs the appropriate validation.
pub struct TraceValidator<B: Backend> {
    model_path: PathBuf,
    context: ModelContext<B>,
}

impl<B: Backend> TraceValidator<B> {
    /// Create a new trace validator for the given model path.
    ///
    /// Automatically detects the model type from config.json.
    pub fn new(model_path: &Path) -> Result<Self, Error> {
        if !model_path.exists() {
            return Err(Error::ModelFolderNotFound);
        }

        let config_path = model_path.join("config.json");
        let config_file = File::open(&config_path)?;
        let model_config: AnyModelConfig = serde_json::from_reader(BufReader::new(config_file))?;

        let context = match model_config {
            AnyModelConfig::ClassifierModelConfig(model_config) => {
                let layer_trace_requirements = model_config
                    .classifier_config
                    .transformer_config
                    .layer_configs
                    .iter()
                    .map(|layer_config| LayerTraceRequirements {
                        post_mixer_norm: layer_config.post_mixer_norm_config.is_some(),
                        post_mlp_norm: layer_config.post_mlp_norm_config.is_some(),
                    })
                    .collect();
                ModelContext::Classifier {
                    classifier: Classifier::new(model_path, &model_config)?,
                    layer_trace_requirements,
                }
            },
            AnyModelConfig::LanguageModelConfig(model_config) => {
                let layer_trace_requirements = model_config
                    .decoder_config
                    .transformer_config
                    .layer_configs
                    .iter()
                    .map(|layer_config| LayerTraceRequirements {
                        post_mixer_norm: layer_config.post_mixer_norm_config.is_some(),
                        post_mlp_norm: layer_config.post_mlp_norm_config.is_some(),
                    })
                    .collect();
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
                ModelContext::LanguageModelGenerator {
                    context: llm_context,
                    layer_trace_requirements,
                }
            },
            AnyModelConfig::TTSModelConfig(_) => {
                return Err(Error::InvalidModelConfig("TTS trace validation is not supported".to_string()));
            },
        };

        Ok(Self {
            model_path: model_path.to_path_buf(),
            context,
        })
    }

    /// Run trace validation and return results.
    pub fn run(&mut self) -> Result<TracerValidationResults, Error> {
        let traces_path = self.model_path.join("traces.safetensors");
        match &mut self.context {
            ModelContext::LanguageModelGenerator {
                context,
                layer_trace_requirements,
            } => Self::run_llm_validation(context, layer_trace_requirements, &traces_path),
            ModelContext::Classifier {
                classifier,
                layer_trace_requirements,
            } => Self::run_classifier_validation(classifier, layer_trace_requirements, &traces_path),
        }
    }

    // ========================================================================
    // LLM Validation
    // ========================================================================

    fn run_llm_validation(
        ctx: &LanguageModelGeneratorContext<B>,
        layer_trace_requirements: &[LayerTraceRequirements],
        traces_path: &Path,
    ) -> Result<TracerValidationResults, Error> {
        let traces_file = File::open(traces_path).map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let (_header_len, traces_metadata) =
            read_safetensors_metadata(&traces_file).map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let trace_shapes: HashMap<String, Vec<usize>> =
            traces_metadata.tensors.iter().map(|(name, tensor)| (name.clone(), tensor.shape.clone())).collect();
        let token_shape = Self::trace_shape(&trace_shapes, "activation_trace.token_ids");
        let position_shape = Self::trace_shape(&trace_shapes, "activation_trace.token_positions");
        let suffix_length = Self::trace_token_count("activation_trace.token_ids", token_shape);
        assert_eq!(
            Self::trace_token_count("activation_trace.token_positions", position_shape),
            suffix_length,
            "trace token ids and token positions must have the same suffix length"
        );

        let traces_loader = ParameterLoader::new(&traces_file, ctx.context.as_ref())
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let traces_view = traces_loader.tree();

        let token_ids = Self::load_array_as_vec::<i32, u64>(&traces_view, &trace_shapes, "activation_trace.token_ids");
        let token_positions =
            Self::load_array_as_vec::<i32, usize>(&traces_view, &trace_shapes, "activation_trace.token_positions");
        let token_inputs = TokenInputs::new_llm(
            ctx.context.as_ref(),
            &ctx.model_shape,
            &token_ids,
            None,
            &token_positions,
            None,
            /*sampling_start=*/ 0,
            /*sampling_length=*/ suffix_length,
        );
        let mut traces = ActivationTrace::new_llm(ctx.context.as_ref(), &ctx.model_shape, suffix_length);

        let mut encoder =
            Encoder::<B>::new(ctx.context.as_ref()).map_err(|e| Error::UnableToCreateCommandBuffer(e.into()))?;
        {
            let mut cache_layers = ctx.cache_layers.borrow_mut();
            cache_layers.prepare_for_forward_pass(ctx.context.as_ref(), suffix_length);
            let decoder_arguments = token_inputs.decoder_arguments(
                Some(&mut *cache_layers),
                suffix_length,
                /*sampling_start=*/ 0,
                /*sampling_length=*/ suffix_length,
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
                .map_err(|e| Error::EncodeFailed(Box::new(e)))?;
        }
        let pending = encoder.end_encoding().submit();
        pending.wait_until_completed().map_err(|e| Error::CommandBufferFailed(Box::new(e)))?;

        // Common layer validation
        let mut results = Self::validate_layer_traces(
            &traces,
            &traces_view,
            &trace_shapes,
            ctx.model_shape.data_type,
            layer_trace_requirements,
        );

        // LLM-specific state validation
        let cache = ctx.cache_layers.borrow();
        for index in 0..ctx.model_shape.num_layers {
            let layer = cache.layer_for_logical_layer(index);
            if let Some(kv) = layer.as_transformer() {
                Self::validate_optional_kv_cache_state(
                    ctx,
                    &mut results,
                    &traces_view,
                    &trace_shapes,
                    kv,
                    &format!("activation_trace.layer_results.{}.activation_trace.state", index),
                );
                Self::validate_optional_kv_cache_state(
                    ctx,
                    &mut results,
                    &traces_view,
                    &trace_shapes,
                    kv,
                    &format!("activation_trace.layer_results.{}.updated_state", index),
                );
            } else if let Some(ssm) = layer.as_state_space() {
                let conv_path = format!("activation_trace.layer_results.{}.updated_state.conv_state", index);
                let expected =
                    Self::read_required_array(&traces_view, &trace_shapes, &conv_path, ctx.model_shape.data_type);
                let conv_state = ssm.conv_state.as_ref().expect("SSM trace requires a produced conv state");
                results.push(TracerValidationResult {
                    name: conv_path,
                    metrics: Self::validate_allocation(
                        ctx.model_shape.data_type,
                        &expected,
                        conv_state,
                        &ssm.conv_shape,
                        Some(ArrayTransform::SsmConvState),
                    ),
                });

                let ssm_path = format!("activation_trace.layer_results.{}.updated_state.ssm_state", index);
                let expected =
                    Self::read_required_array(&traces_view, &trace_shapes, &ssm_path, ctx.model_shape.data_type);
                results.push(TracerValidationResult {
                    name: ssm_path,
                    metrics: Self::validate_allocation(
                        ctx.model_shape.data_type,
                        &expected,
                        &ssm.ssm_state,
                        &ssm.ssm_shape,
                        None,
                    ),
                });
            } else if let Some(delta) = layer.as_delta_net() {
                let conv_path = format!("activation_trace.layer_results.{}.updated_state.conv_state", index);
                let expected =
                    Self::read_required_array(&traces_view, &trace_shapes, &conv_path, ctx.model_shape.data_type);
                results.push(TracerValidationResult {
                    name: conv_path,
                    metrics: Self::validate_allocation(
                        ctx.model_shape.data_type,
                        &expected,
                        &delta.conv_state,
                        &delta.conv_shape,
                        Some(ArrayTransform::SsmConvState),
                    ),
                });

                let ssm_path = format!("activation_trace.layer_results.{}.updated_state.ssm_state", index);
                let expected =
                    Self::read_required_array(&traces_view, &trace_shapes, &ssm_path, ctx.model_shape.data_type);
                results.push(TracerValidationResult {
                    name: ssm_path,
                    metrics: Self::validate_allocation(
                        ctx.model_shape.data_type,
                        &expected,
                        &delta.ssm_state,
                        &delta.ssm_shape,
                        None,
                    ),
                });
            } else if let Some(short_conv) = layer.as_short_conv() {
                let conv_path = format!("activation_trace.layer_results.{}.updated_state.conv_state", index);
                let expected =
                    Self::read_required_array(&traces_view, &trace_shapes, &conv_path, ctx.model_shape.data_type);
                results.push(TracerValidationResult {
                    name: conv_path,
                    metrics: Self::validate_allocation(
                        ctx.model_shape.data_type,
                        &expected,
                        &short_conv.conv_state,
                        &short_conv.conv_shape,
                        Some(ArrayTransform::SsmConvState),
                    ),
                });
            } else {
                panic!("Unsupported cache layer type at layer {index}");
            }
        }
        for (compact_index, (_layer_index, layer)) in cache.iter_layers().enumerate() {
            if let Some(kv) = layer.as_transformer() {
                Self::validate_optional_kv_cache_state(
                    ctx,
                    &mut results,
                    &traces_view,
                    &trace_shapes,
                    kv,
                    &format!("updated_state.{}", compact_index),
                );
            }
        }
        results.extend(Self::validate_rope_traces(ctx, &traces_view, &trace_shapes, &token_positions));

        // LLM-specific: Token comparison
        let expected_logits =
            Self::read_required_array(&traces_view, &trace_shapes, "logits", ctx.model_shape.data_type);
        let expected_tokens = Self::get_tokens_from_logits(&expected_logits);
        let produced_tokens = Self::get_tokens_from_logits(&traces.logits);
        let tokens_violation_indices = expected_tokens
            .iter()
            .zip(produced_tokens.iter())
            .enumerate()
            .filter_map(|(i, (a, b))| {
                if a != b {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        traces_view.assert_all_tensors_validated().unwrap_or_else(|error| {
            panic!("Trace contains tensors that Uzu tracer did not validate: {error}");
        });

        Ok(TracerValidationResults {
            suffix_length,
            results,
            tokens_violation_indices,
        })
    }

    fn validate_optional_kv_cache_state(
        ctx: &LanguageModelGeneratorContext<B>,
        results: &mut Vec<TracerValidationResult>,
        traces_view: &ParameterTree<B>,
        trace_shapes: &HashMap<String, Vec<usize>>,
        kv: &dyn KVCacheLayerTrait<B>,
        prefix: &str,
    ) {
        let shape = kv.shape();
        let size = shape.iter().product::<usize>() * ctx.model_shape.data_type.size_in_bytes();
        for field in ["keys", "values"] {
            let path = format!("{prefix}.{field}");
            if !trace_shapes.contains_key(&path) {
                continue;
            }
            let allocation = Self::read_kv_buffer(ctx, kv, field, size);
            let expected = Self::read_required_array(traces_view, trace_shapes, &path, ctx.model_shape.data_type);
            results.push(TracerValidationResult {
                name: path,
                metrics: Self::validate_allocation(
                    ctx.model_shape.data_type,
                    &expected,
                    &allocation,
                    &shape,
                    Some(ArrayTransform::KVCacheSlice {
                        start: kv.prefix_segment_length(),
                    }),
                ),
            });
        }
    }

    fn read_kv_buffer(
        ctx: &LanguageModelGeneratorContext<B>,
        kv: &dyn KVCacheLayerTrait<B>,
        field: &str,
        size: usize,
    ) -> Allocation<B> {
        if let Some(layer) = kv.as_any().downcast_ref::<KVCacheLayer<B, B::SparseBuffer>>() {
            let buffer = if field == "keys" {
                &layer.keys
            } else {
                &layer.values
            };
            common::helpers::sparse_buffer_read_allocation(ctx.context.as_ref(), buffer, size)
        } else if let Some(layer) = kv.as_any().downcast_ref::<KVCacheLayer<B, B::DenseBuffer>>() {
            let buffer = if field == "keys" {
                &layer.keys
            } else {
                &layer.values
            };
            let mut allocation = ctx
                .context
                .create_allocation(size, AllocationType::Global)
                .expect("Failed to create KV cache read allocation");
            let mut encoder = Encoder::<B>::new(ctx.context.as_ref()).expect("Failed to create KV cache read encoder");
            encoder.encode_copy(buffer, 0..size, &mut allocation, 0..size);
            encoder.end_encoding().submit().wait_until_completed().expect("Failed to read KV cache buffer");
            allocation
        } else {
            panic!("Unexpected KV cache buffer type")
        }
    }

    fn validate_rope_traces(
        ctx: &LanguageModelGeneratorContext<B>,
        traces_view: &ParameterTree<B>,
        trace_shapes: &HashMap<String, Vec<usize>>,
        token_positions: &[usize],
    ) -> Vec<TracerValidationResult> {
        let mut results = Vec::new();
        for (rope_index, rope) in ctx.shared_buffers.rope_buffers.iter().enumerate() {
            for (name, allocation) in [("cosines", &rope.cosines), ("sines", &rope.sines)] {
                Self::validate_optional_rope_tensor(
                    &mut results,
                    traces_view,
                    trace_shapes,
                    &format!("activation_trace.rope_embeddings.{rope_index}.{name}"),
                    allocation,
                    &[rope.max_sequence_length(), rope.dim()],
                    token_positions,
                    ctx.model_shape.rope_data_type,
                );
            }
        }

        for layer_index in 0..ctx.model_shape.num_layers {
            let Some(rope) = ctx.shared_buffers.rope_buffers_for_layer(layer_index) else {
                continue;
            };
            for (name, allocation) in [("cosines", &rope.cosines), ("sines", &rope.sines)] {
                Self::validate_optional_rope_tensor(
                    &mut results,
                    traces_view,
                    trace_shapes,
                    &format!(
                        "activation_trace.layer_results.{layer_index}.activation_trace.positional_embeddings.{name}"
                    ),
                    allocation,
                    &[rope.max_sequence_length(), rope.dim()],
                    token_positions,
                    ctx.model_shape.rope_data_type,
                );
            }
        }
        results
    }

    fn validate_optional_rope_tensor(
        results: &mut Vec<TracerValidationResult>,
        traces_view: &ParameterTree<B>,
        trace_shapes: &HashMap<String, Vec<usize>>,
        path: &str,
        allocation: &Allocation<B>,
        shape: &[usize],
        token_positions: &[usize],
        data_type: DataType,
    ) {
        if !trace_shapes.contains_key(path) {
            return;
        }
        let expected = Self::read_required_array(traces_view, trace_shapes, path, data_type);
        results.push(TracerValidationResult {
            name: path.to_string(),
            metrics: Self::validate_allocation(
                data_type,
                &expected,
                allocation,
                shape,
                Some(ArrayTransform::PositionSlice(token_positions.to_vec())),
            ),
        });
    }

    // ========================================================================
    // Classifier Validation
    // ========================================================================

    fn run_classifier_validation(
        classifier: &mut Classifier<B>,
        layer_trace_requirements: &[LayerTraceRequirements],
        traces_path: &Path,
    ) -> Result<TracerValidationResults, Error> {
        let traces_file = File::open(traces_path).map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let (_header_len, traces_metadata) =
            read_safetensors_metadata(&traces_file).map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let trace_shapes: HashMap<String, Vec<usize>> =
            traces_metadata.tensors.iter().map(|(name, tensor)| (name.clone(), tensor.shape.clone())).collect();
        let context = classifier.context.context.clone();
        let traces_loader = ParameterLoader::new(&traces_file, context.as_ref())
            .map_err(|error| Error::UnableToLoadWeights(Box::new(error)))?;
        let traces_view = traces_loader.tree();

        let token_shape = Self::trace_shape(&trace_shapes, "activation_trace.token_ids");
        let position_shape = Self::trace_shape(&trace_shapes, "activation_trace.token_positions");
        let suffix_length = Self::trace_token_count("activation_trace.token_ids", token_shape);
        assert_eq!(
            Self::trace_token_count("activation_trace.token_positions", position_shape),
            suffix_length,
            "trace token ids and token positions must have the same suffix length"
        );

        let token_ids = Self::load_array_as_vec::<i32, u64>(&traces_view, &trace_shapes, "activation_trace.token_ids");
        let token_positions =
            Self::load_array_as_vec::<i32, usize>(&traces_view, &trace_shapes, "activation_trace.token_positions");

        let (_logits, traces) = classifier.forward_pass_with_traces(&token_ids, &token_positions)?;

        // Common layer validation
        let mut results = Self::validate_layer_traces(
            &traces,
            &traces_view,
            &trace_shapes,
            classifier.context.model_shape.data_type,
            layer_trace_requirements,
        );

        // Classifier-specific: embedding_norm, output_pooling
        let classifier_results = Self::validate_classifier_traces(
            &traces,
            &traces_view,
            &trace_shapes,
            classifier.context.model_shape.data_type,
        );
        results.extend(classifier_results);
        traces_view.assert_all_tensors_validated().unwrap_or_else(|error| {
            panic!("Trace contains tensors that Uzu tracer did not validate: {error}");
        });

        Ok(TracerValidationResults {
            suffix_length,
            results,
            tokens_violation_indices: Vec::new(), // Classifiers don't compare tokens
        })
    }

    // ========================================================================
    // Common Validation Helpers
    // ========================================================================

    fn validate_layer_traces(
        traces: &ActivationTrace<B>,
        traces_view: &ParameterTree<B>,
        trace_shapes: &HashMap<String, Vec<usize>>,
        data_type: DataType,
        layer_trace_requirements: &[LayerTraceRequirements],
    ) -> Vec<TracerValidationResult> {
        let mut results = Vec::new();

        assert_eq!(
            traces.layer_results.len(),
            layer_trace_requirements.len(),
            "trace layer count must match model layer count"
        );

        let validate = |path: &str, array: &Array<B>| -> TracerValidationResult {
            let expected = Self::read_required_array(traces_view, trace_shapes, path, data_type);
            TracerValidationResult {
                name: path.to_string(),
                metrics: Self::validate_allocation(data_type, &expected, array.allocation(), array.shape(), None),
            }
        };

        for (index, layer_traces) in traces.layer_results.iter().enumerate() {
            let requirements = layer_trace_requirements[index];
            let path = |suffix: &str| -> String {
                format!("activation_trace.layer_results.{}.activation_trace.{}", index, suffix)
            };

            results.push(validate(&path("inputs"), &layer_traces.inputs));
            results.push(validate(&path("pre_mixer_norm"), &layer_traces.pre_attention_norm));
            results.push(validate(&path("mixer"), &layer_traces.attention));
            if requirements.post_mixer_norm {
                results.push(validate(&path("post_mixer_norm"), &layer_traces.post_attention_norm));
            }
            results.push(validate(&path("mlp_inputs"), &layer_traces.mlp_inputs));
            results.push(validate(&path("pre_mlp_norm"), &layer_traces.pre_mlp_norm));
            results.push(validate(&path("mlp"), &layer_traces.mlp));
            if requirements.post_mlp_norm {
                results.push(validate(&path("post_mlp_norm"), &layer_traces.post_mlp_norm));
            }

            let outputs_path = format!("activation_trace.layer_results.{}.outputs", index);
            results.push(validate(&outputs_path, &layer_traces.outputs));
        }

        // Output norm (common to all models)
        results.push(validate("activation_trace.output_norm", &traces.output_norm));

        // Logits live on DecoderResult/ClassifierResult, not inside activation_trace.
        results.push(validate("logits", &traces.logits));

        results
    }

    fn validate_classifier_traces(
        traces: &ActivationTrace<B>,
        traces_view: &ParameterTree<B>,
        trace_shapes: &HashMap<String, Vec<usize>>,
        data_type: DataType,
    ) -> Vec<TracerValidationResult> {
        let mut results = Vec::new();

        let embedding_norm =
            traces.embedding_norm.as_ref().expect("classifier traces must include produced embedding norm output");
        let expected =
            Self::read_required_array(traces_view, trace_shapes, "activation_trace.embedding_norm_output", data_type);
        results.push(TracerValidationResult {
            name: "activation_trace.embedding_norm_output".to_string(),
            metrics: Self::validate_allocation(
                data_type,
                &expected,
                embedding_norm.allocation(),
                embedding_norm.shape(),
                None,
            ),
        });

        if let Some(output_pooling) = &traces.output_pooling {
            let expected =
                Self::read_required_array(traces_view, trace_shapes, "activation_trace.output_pooling", data_type);
            results.push(TracerValidationResult {
                name: "activation_trace.output_pooling".to_string(),
                metrics: Self::validate_allocation(
                    data_type,
                    &expected,
                    output_pooling.allocation(),
                    output_pooling.shape(),
                    None,
                ),
            });
        } else {
            panic!("classifier traces must include produced output pooling");
        };

        results
    }

    // ========================================================================
    // Allocation Validation
    // ========================================================================

    fn validate_allocation(
        data_type: DataType,
        expected_array: &Array<B>,
        produced_allocation: &Allocation<B>,
        produced_shape: &[usize],
        transform: Option<ArrayTransform>,
    ) -> TracerValidationMetrics {
        match data_type {
            DataType::F16 => {
                Self::validate_allocation_of_type::<f16>(expected_array, produced_allocation, produced_shape, transform)
            },
            DataType::BF16 => Self::validate_allocation_of_type::<bf16>(
                expected_array,
                produced_allocation,
                produced_shape,
                transform,
            ),
            DataType::F32 => {
                Self::validate_allocation_of_type::<f32>(expected_array, produced_allocation, produced_shape, transform)
            },
            _ => panic!("Unsupported data type: {:?}", data_type),
        }
    }

    fn validate_allocation_of_type<Precision: ArrayElement>(
        expected_array: &Array<B>,
        produced_allocation: &Allocation<B>,
        produced_shape: &[usize],
        transform: Option<ArrayTransform>,
    ) -> TracerValidationMetrics {
        let produced = produced_allocation.copyout::<Precision>();
        Self::validate_allocation_data_of_type(expected_array, &produced, produced_shape, transform)
    }

    fn validate_allocation_data_of_type<Precision: ArrayElement>(
        expected_array: &Array<B>,
        produced_slice: &[Precision],
        produced_shape: &[usize],
        transform: Option<ArrayTransform>,
    ) -> TracerValidationMetrics {
        let expected_view = expected_array.as_view::<Precision>();
        let produced_view = ndarray::ArrayView::from_shape(IxDyn(produced_shape), produced_slice)
            .expect("Failed to reshape allocation");

        let (expected_data, mut produced_data) = match transform {
            Some(ArrayTransform::KVCacheSlice {
                start,
            }) => {
                let expected_shape = expected_view.shape();
                let expected_tokens = expected_view.shape()[1];
                let sliced = produced_view.slice(s![start..start + expected_tokens, .., ..]);
                let reshaped = sliced
                    .into_owned()
                    .to_shape(IxDyn(expected_shape))
                    .expect("Failed to reshape KV cache slice")
                    .to_owned();
                (expected_view.to_owned(), reshaped)
            },
            Some(ArrayTransform::PositionSlice(positions)) => {
                let produced_shape = produced_view.shape();
                let [max_sequence_length, dim] = produced_shape else {
                    panic!("PositionSlice requires produced shape [sequence, dim], got {produced_shape:?}");
                };
                for &position in &positions {
                    assert!(
                        position < *max_sequence_length,
                        "traced token position {position} exceeds produced sequence length {max_sequence_length}"
                    );
                }

                let expected_shape = expected_view.shape();
                let gathered_shape = if expected_shape == [1, positions.len(), *dim] {
                    vec![1, positions.len(), *dim]
                } else {
                    vec![positions.len(), *dim]
                };
                let mut gathered = Vec::with_capacity(positions.len() * *dim);
                for position in positions {
                    gathered.extend(produced_view.slice(s![position, ..]).iter().copied());
                }
                let produced_data = ndarray::ArrayD::from_shape_vec(IxDyn(&gathered_shape), gathered)
                    .expect("Failed to reshape gathered position slice");
                (expected_view.to_owned(), produced_data)
            },
            Some(ArrayTransform::SsmConvState) => {
                let produced_shape = produced_view.shape();
                let history_len = produced_shape[1];
                let dim = produced_shape[0];

                let permuted = expected_view.permuted_axes(IxDyn(&[0, 2, 1]));
                let total_time = permuted.shape()[2];
                let start = total_time.saturating_sub(history_len);
                let sliced = permuted.slice(s![.., .., start..]);

                let reshaped_expected = sliced
                    .into_owned()
                    .to_shape(IxDyn(&[dim, history_len]))
                    .expect("Failed to reshape SSM conv state slice")
                    .to_owned();

                (reshaped_expected, produced_view.to_owned())
            },
            None => (expected_view.to_owned(), produced_view.to_owned()),
        };

        let expected_shape = expected_data.shape().to_vec();
        let produced_shape = produced_data.shape().to_vec();

        if expected_shape != produced_shape {
            if expected_shape.len() == produced_shape.len() + 1
                && expected_shape.first() == Some(&1)
                && expected_shape[1..] == produced_shape[..]
            {
                produced_data = produced_data
                    .to_shape(IxDyn(&expected_shape))
                    .expect("Failed to reshape produced data to trace shape")
                    .to_owned();
            } else {
                panic!(
                    "Shape mismatch: expected trace shape {expected_shape:?}, produced Uzu shape {produced_shape:?}"
                );
            }
        }

        if expected_data.shape() != produced_data.shape() {
            panic!(
                "Shape mismatch after alignment: expected {:?}, produced {:?}",
                expected_data.shape(),
                produced_data.shape()
            );
        }

        let reference: Vec<f32> = expected_data
            .iter()
            .map(|value| NumCast::from(*value).expect("Failed to cast reference trace value"))
            .collect();
        let result: Vec<f32> = produced_data
            .iter()
            .map(|value| NumCast::from(*value).expect("Failed to cast produced trace value"))
            .collect();

        let (atol, rtol, allowed_voilations_tol, rms_rtol) = match expected_array.data_type() {
            DataType::BF16 => (0.04, 0.06, 0.03, 0.035),
            _ => (0.01, 0.03, 0.01, 0.01),
        };

        Self::compare_arrays(
            &reference,
            expected_data.shape().to_vec(),
            &result,
            produced_data.shape().to_vec(),
            atol,
            rtol,
            allowed_voilations_tol,
            rms_rtol,
        )
    }

    fn compare_arrays(
        reference: &[f32],
        reference_shape: Vec<usize>,
        result: &[f32],
        result_shape: Vec<usize>,
        atol: f32,
        rtol: f32,
        fraction_of_allowed_violations: f32,
        rms_rtol: f32,
    ) -> TracerValidationMetrics {
        assert_eq!(result.len(), reference.len());
        if reference.is_empty() {
            return TracerValidationMetrics {
                atol,
                rtol,
                rms_rtol,
                fraction_of_allowed_violations,
                reference_shape,
                result_shape,
                num_violations: 0,
                max_allowed_violations: 0,
                max_err_idx: 0,
                max_err: 0.0,
                max_err_rel: 0.0,
                max_err_reference_value: 0.0,
                rms_diff: 0.0,
                rms_result: 0.0,
                rms_reference: 0.0,
                rel_rms_reference: 0.0,
                diff_max: 0.0,
                diff_avg: 0.0,
                result_nan: false,
            };
        }

        let mut num_violations = 0;
        let mut max_err = 0.0f32;
        let mut max_err_idx = 0;
        let mut max_err_rel = 0.0f32;
        let mut max_err_reference_value = 0.0f32;
        let mut sum_sq_diff = 0.0f32;
        let mut sum_sq_result = 0.0f32;
        let mut sum_sq_reference = 0.0f32;
        let mut diff_sum = 0.0f32;
        let mut diff_max = 0.0f32;
        let mut result_nan = false;

        for (i, (&exp, &prod)) in reference.iter().zip(result.iter()).enumerate() {
            if prod.is_nan() {
                result_nan = true;
            }

            let abs_diff = (exp - prod).abs();
            let rel_diff = if exp.abs() > 1e-8 {
                abs_diff / exp.abs()
            } else {
                abs_diff
            };

            diff_sum += abs_diff;
            diff_max = diff_max.max(abs_diff);
            sum_sq_diff += abs_diff * abs_diff;
            sum_sq_result += prod * prod;
            sum_sq_reference += exp * exp;

            if abs_diff > atol && rel_diff > rtol {
                num_violations += 1;
            }

            if abs_diff > max_err {
                max_err = abs_diff;
                max_err_idx = i;
                max_err_rel = rel_diff;
                max_err_reference_value = exp;
            }
        }

        let n = reference.len() as f32;
        let rms_diff = (sum_sq_diff / n).sqrt();
        let rms_result = (sum_sq_result / n).sqrt();
        let rms_reference = (sum_sq_reference / n).sqrt();
        let rel_rms_reference = if rms_reference > 1e-8 {
            rms_diff / rms_reference
        } else {
            rms_diff
        };
        let diff_avg = diff_sum / n;

        let max_allowed_violations = (fraction_of_allowed_violations * n).ceil() as usize;

        TracerValidationMetrics {
            atol,
            rtol,
            rms_rtol,
            fraction_of_allowed_violations,
            reference_shape,
            result_shape,
            num_violations,
            max_allowed_violations,
            max_err_idx,
            max_err,
            max_err_rel,
            max_err_reference_value,
            rms_diff,
            rms_result,
            rms_reference,
            rel_rms_reference,
            diff_max,
            diff_avg,
            result_nan,
        }
    }

    // ========================================================================
    // Utility Functions
    // ========================================================================

    fn load_array_as_vec<SourcePrecision: ArrayElement, TargetPrecision: NumCast>(
        traces_view: &ParameterTree<B>,
        trace_shapes: &HashMap<String, Vec<usize>>,
        name: &str,
    ) -> Vec<TargetPrecision> {
        let expected_shape = Self::trace_shape(trace_shapes, name);
        let leaf = traces_view
            .leaf(name)
            .unwrap_or_else(|error| panic!("Missing required trace tensor {name}: {error}"))
            .validate(expected_shape, SourcePrecision::data_type())
            .unwrap_or_else(|error| panic!("Invalid trace tensor {name}: {error}"));
        let slice = leaf.read_slice::<SourcePrecision>().unwrap_or_else(|error| {
            panic!("Failed to read trace tensor {name}: {error}");
        });
        slice.iter().map(|x| NumCast::from(*x).expect("Failed to cast trace tensor value")).collect()
    }

    fn trace_shape<'shape>(
        trace_shapes: &'shape HashMap<String, Vec<usize>>,
        name: &str,
    ) -> &'shape [usize] {
        trace_shapes.get(name).unwrap_or_else(|| panic!("Missing required trace tensor {name}"))
    }

    fn trace_token_count(
        name: &str,
        shape: &[usize],
    ) -> usize {
        let [batch_size, suffix_length] = shape else {
            panic!("Trace tensor {name} must have shape [1, suffix_tokens], got {shape:?}");
        };
        assert_eq!(*batch_size, 1, "Trace tensor {name} must have batch size 1, got {shape:?}");
        *suffix_length
    }

    fn read_required_array(
        traces_view: &ParameterTree<B>,
        trace_shapes: &HashMap<String, Vec<usize>>,
        name: &str,
        data_type: DataType,
    ) -> Array<B> {
        let expected_shape = Self::trace_shape(trace_shapes, name);
        Self::read_array(traces_view, name, expected_shape, data_type)
            .unwrap_or_else(|error| panic!("Invalid trace tensor {name}: {error}"))
    }

    fn read_array(
        traces_view: &ParameterTree<B>,
        name: &str,
        expected_shape: &[usize],
        data_type: DataType,
    ) -> Result<Array<B>, ParameterLoaderError<B>> {
        traces_view.leaf(name)?.validate(expected_shape, data_type)?.read_array()
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
        let suffix_length = Self::trace_token_count("activation_trace.token_ids", token_shape);
        assert!(suffix_length > 0, "trace must contain at least one token");
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

    fn get_tokens_from_logits(logits: &Array<B>) -> Vec<u64> {
        let data_type = logits.data_type();
        match data_type {
            DataType::F16 => sample_argmax(logits.as_view::<f16>()),
            DataType::BF16 => sample_argmax(logits.as_view::<bf16>()),
            DataType::F32 => sample_argmax(logits.as_view::<f32>()),
            _ => panic!("Unsupported data type: {:?}", data_type),
        }
    }
}
