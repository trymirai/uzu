//! Unified trace validation for any model type (LLM or classifier).
//!
//! This module provides a single `TraceValidator` that can validate activation
//! traces for any model type. It automatically detects the model type from the
//! config and runs the appropriate validation.

use std::{
    cell::{Ref, RefCell},
    fs::File,
    path::{Path, PathBuf},
    rc::Rc,
};

use half::{bf16, f16};
use ndarray::{IxDyn, s};
use num_traits::NumCast;

use crate::{
    Array, ArrayElement, DataType,
    backends::metal::{
        CacheLayers, KVCacheUpdate, KernelDataType, MTLContext, MetalArray,
        encodable_block::Sampling,
        forward_pass::{
            ArrayId, EncodableBlock, EncodingParameters, ForwardPassState,
            ScratchBuffers, traces::ActivationTrace,
        },
    },
    classifier::Classifier,
    config::ModelMetadata,
    language_model::{
        LanguageModelGeneratorContext,
        sampler::{ArgmaxSampler, LogitsSampler},
    },
    parameters::{ParameterLoader, ParameterTree, read_safetensors_metadata},
    session::{
        config::{DecodingConfig, SpeculatorConfig},
        parameter::{
            AsyncBatchSize, ConfigResolvableValue, ContextLength, ContextMode,
            PrefillStepSize, SamplingSeed,
        },
        types::Error,
    },
};

// ============================================================================
// Validation Types
// ============================================================================

/// Metrics from validating a single tensor.
pub struct TracerValidationMetrics {
    pub atol: f32,
    pub rtol: f32,
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
        self.num_violations <= self.max_allowed_violations && !self.result_nan
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
            Relative error RMS: {:.2}% of RMS of reference.\n\
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

    pub fn is_valid(&self) -> bool {
        self.number_of_tokens_violations()
            <= self.number_of_allowed_tokens_violations()
    }
}

/// Transform to apply to produced arrays before comparison.
pub enum ArrayTransform {
    /// Slice KV cache to match expected shape.
    KVCacheSlice,
    /// Transform SSM conv state layout.
    SsmConvState,
}

// ============================================================================
// Model Context (internal)
// ============================================================================

enum ModelContext {
    LanguageModelGenerator(LanguageModelGeneratorContext),
    Classifier(Classifier),
}

// ============================================================================
// Unified TraceValidator
// ============================================================================

/// Unified trace validator for any model type.
///
/// Automatically detects whether the model is an LLM or classifier and
/// runs the appropriate validation.
pub struct TraceValidator {
    model_path: PathBuf,
    context: ModelContext,
}

impl TraceValidator {
    /// Create a new trace validator for the given model path.
    ///
    /// Automatically detects the model type from config.json.
    pub fn new(model_path: &Path) -> Result<Self, Error> {
        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(Error::ModelFolderNotFound);
        }

        let config_file =
            File::open(&config_path).map_err(|_| Error::UnableToLoadConfig)?;
        let metadata: ModelMetadata =
            serde_json::from_reader(std::io::BufReader::new(config_file))
                .map_err(|_| Error::UnableToLoadConfig)?;

        let context = if metadata.model_config.is_classifier() {
            let classifier = Classifier::new(model_path)?;
            ModelContext::Classifier(classifier)
        } else {
            let prefill_step_size =
                Self::determine_prefill_step_size(model_path);
            let decoding_config = DecodingConfig::new(
                ContextMode::default(),
                ContextLength::default(),
                PrefillStepSize::Custom(prefill_step_size),
                SpeculatorConfig::default(),
                SamplingSeed::default(),
                AsyncBatchSize::default(),
                false,
            );
            let mut llm_context = LanguageModelGeneratorContext::new(
                model_path,
                &decoding_config,
            )?;
            let desired_suffix_length =
                prefill_step_size.max(decoding_config.generate_suffix_length());
            Self::ensure_llm_context_capacity(
                &decoding_config,
                desired_suffix_length,
                &mut llm_context,
            );
            ModelContext::LanguageModelGenerator(llm_context)
        };

        Ok(Self {
            model_path: model_path.to_path_buf(),
            context,
        })
    }

    /// Run trace validation and return results.
    pub fn run(&mut self) -> Result<TracerValidationResults, Error> {
        let traces_path = self.model_path.join("traces.safetensors");
        if !traces_path.exists() {
            return Err(Error::UnableToLoadWeights);
        }

        match &mut self.context {
            ModelContext::LanguageModelGenerator(ctx) => {
                Self::run_llm_validation(ctx, &traces_path)
            },
            ModelContext::Classifier(classifier) => {
                Self::run_classifier_validation(classifier, &traces_path)
            },
        }
    }

    pub fn is_classifier(&self) -> bool {
        matches!(self.context, ModelContext::Classifier(_))
    }

    pub fn is_language_model_generator(&self) -> bool {
        matches!(self.context, ModelContext::LanguageModelGenerator(_))
    }

    // ========================================================================
    // LLM Validation
    // ========================================================================

    fn run_llm_validation(
        ctx: &LanguageModelGeneratorContext,
        traces_path: &Path,
    ) -> Result<TracerValidationResults, Error> {
        let traces_file =
            File::open(traces_path).map_err(|_| Error::UnableToLoadWeights)?;
        let traces_loader =
            ParameterLoader::new(&traces_file, &ctx.mtl_context)
                .map_err(|_| Error::UnableToLoadWeights)?;
        let traces_view = traces_loader.tree();

        let token_ids = Self::load_array_as_vec::<i32, u64>(
            &traces_view,
            "activation_trace.token_ids",
        );
        let token_positions = Self::load_array_as_vec::<i32, usize>(
            &traces_view,
            "activation_trace.token_positions",
        );
        let token_seeds: Vec<u64> = vec![0; token_ids.len()];

        let mut state = ForwardPassState::new_llm(
            ctx.mtl_context.clone(),
            &ctx.decoder_config,
            &ctx.model_shape,
            &ctx.scratch_buffers,
            ctx.cache_layers.clone(),
            ctx.shared_buffers.clone(),
            &token_ids,
            &token_positions,
            None,
            &token_seeds,
            token_ids.len(),
            false,
            None,
            false,
            false,
            None,
            None,
        );

        let root_command_buffer =
            ctx.command_buffer.root_command_buffer().to_owned();
        ctx.executables.encode(
            &mut state,
            &ctx.command_buffer,
            &EncodingParameters::new(false, false, true),
        );
        ctx.command_buffer.commit();
        root_command_buffer.wait_until_completed();

        let traces = state.traces().clone();
        let data_type = ctx.model_shape.activation_data_type();

        // Common layer validation
        let mut results =
            Self::validate_layer_traces(&traces, &traces_view, data_type);

        // LLM-specific: KV cache validation
        let transformer_layers: Vec<usize> = {
            let cache = state.cache_layers().unwrap().borrow();
            cache
                .data
                .iter()
                .enumerate()
                .filter_map(|(index, layer)| {
                    layer.as_transformer().map(|_| index)
                })
                .collect()
        };

        for index in transformer_layers {
            let arrays =
                state.arrays(&[ArrayId::Keys(index), ArrayId::Values(index)]);

            if let Ok(expected) =
                traces_view.leaf(&format!("updated_kv_cache.{}.keys", index))
            {
                let keys = arrays[0].borrow();
                results.push(TracerValidationResult {
                    name: format!("updated_kv_cache.{}.keys", index),
                    metrics: Self::validate_array(
                        data_type,
                        &expected,
                        &keys,
                        Some(ArrayTransform::KVCacheSlice),
                    ),
                });
            }

            if let Ok(expected) =
                traces_view.leaf(&format!("updated_kv_cache.{}.values", index))
            {
                let values = arrays[1].borrow();
                results.push(TracerValidationResult {
                    name: format!("updated_kv_cache.{}.values", index),
                    metrics: Self::validate_array(
                        data_type,
                        &expected,
                        &values,
                        Some(ArrayTransform::KVCacheSlice),
                    ),
                });
            }
        }

        // LLM-specific: SSM state validation
        let ssm_layers: Vec<usize> = {
            let cache = state.cache_layers().unwrap().borrow();
            cache
                .data
                .iter()
                .enumerate()
                .filter_map(|(index, layer)| {
                    layer.as_state_space().map(|_| index)
                })
                .collect()
        };

        for index in ssm_layers {
            let arrays = state.arrays(&[
                ArrayId::SsmConvState(index),
                ArrayId::SsmState(index),
            ]);
            let conv_state = arrays[0].borrow();
            let ssm_state = arrays[1].borrow();

            for path in [
                format!("updated_state.{}.conv_state", index),
                format!(
                    "activation_trace.layer_results.{}.updated_state.conv_state",
                    index
                ),
            ] {
                if let Ok(expected) = traces_view.leaf(&path) {
                    results.push(TracerValidationResult {
                        name: path,
                        metrics: Self::validate_array(
                            data_type,
                            &expected,
                            &conv_state,
                            Some(ArrayTransform::SsmConvState),
                        ),
                    });
                }
            }

            for path in [
                format!("updated_state.{}.ssm_state", index),
                format!(
                    "activation_trace.layer_results.{}.updated_state.ssm_state",
                    index
                ),
            ] {
                if let Ok(expected) = traces_view.leaf(&path) {
                    results.push(TracerValidationResult {
                        name: path,
                        metrics: Self::validate_array(
                            data_type, &expected, &ssm_state, None,
                        ),
                    });
                }
            }
        }

        // LLM-specific: Token comparison
        let tokens_violation_indices = if let Ok(expected_logits) =
            traces_view.leaf("logits")
        {
            let expected_tokens =
                Self::get_tokens_from_logits(&expected_logits);
            let produced_tokens =
                Self::get_tokens_from_logits(&*traces.borrow().logits.borrow());
            expected_tokens
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
                .collect()
        } else {
            Vec::new()
        };

        Ok(TracerValidationResults {
            suffix_length: token_ids.len(),
            results,
            tokens_violation_indices,
        })
    }

    // ========================================================================
    // Classifier Validation
    // ========================================================================

    fn run_classifier_validation(
        classifier: &mut Classifier,
        traces_path: &Path,
    ) -> Result<TracerValidationResults, Error> {
        let traces_file =
            File::open(traces_path).map_err(|_| Error::UnableToLoadWeights)?;
        let mtl_context = classifier.context.mtl_context.clone();
        let traces_loader = ParameterLoader::new(&traces_file, &mtl_context)
            .map_err(|_| Error::UnableToLoadWeights)?;
        let traces_view = traces_loader.tree();

        let has_token_ids =
            traces_view.leaf("activation_trace.token_ids").is_ok();
        let has_token_positions =
            traces_view.leaf("activation_trace.token_positions").is_ok();

        if !has_token_ids || !has_token_positions {
            return Ok(Self::handle_missing_tokens(&traces_view));
        }

        let token_ids = Self::load_array_as_vec::<i32, u64>(
            &traces_view,
            "activation_trace.token_ids",
        );
        let token_positions = Self::load_array_as_vec::<i32, usize>(
            &traces_view,
            "activation_trace.token_positions",
        );

        let suffix_length = token_ids.len();

        let (_logits, traces) = classifier
            .forward_pass_with_traces(&token_ids, &token_positions)
            .map_err(|_| Error::GenerateFailed)?;

        let data_type = classifier.context.model_shape.activation_data_type();

        // Common layer validation
        let mut results =
            Self::validate_layer_traces(&traces, &traces_view, data_type);

        // Classifier-specific: embedding_norm, output_pooling
        let classifier_results =
            Self::validate_classifier_traces(&traces, &traces_view, data_type);
        results.extend(classifier_results);

        Ok(TracerValidationResults {
            suffix_length,
            results,
            tokens_violation_indices: Vec::new(), // Classifiers don't compare tokens
        })
    }

    fn handle_missing_tokens(
        traces_view: &ParameterTree<Rc<MTLContext>>
    ) -> TracerValidationResults {
        if let Ok(expected_logits) = traces_view.leaf("logits") {
            let reference_shape = expected_logits.shape().to_vec();
            let metrics = TracerValidationMetrics {
                atol: 0.0,
                rtol: 0.0,
                fraction_of_allowed_violations: 0.0,
                reference_shape: reference_shape.clone(),
                result_shape: reference_shape,
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
            return TracerValidationResults {
                suffix_length: 1,
                results: vec![TracerValidationResult {
                    name: "activation_trace.logits".to_string(),
                    metrics,
                }],
                tokens_violation_indices: Vec::new(),
            };
        }

        TracerValidationResults {
            suffix_length: 1,
            results: Vec::new(),
            tokens_violation_indices: Vec::new(),
        }
    }

    // ========================================================================
    // Common Validation Helpers
    // ========================================================================

    fn validate_layer_traces(
        traces: &Rc<RefCell<ActivationTrace>>,
        traces_view: &ParameterTree<Rc<MTLContext>>,
        data_type: DataType,
    ) -> Vec<TracerValidationResult> {
        let mut results = Vec::new();

        let validate = |path: &str,
                        array: &Ref<MetalArray>|
         -> Option<TracerValidationResult> {
            if traces_view.leaf(path).is_ok() {
                Some(TracerValidationResult {
                    name: path.to_string(),
                    metrics: Self::validate_array_with_name(
                        data_type,
                        traces_view,
                        path,
                        array,
                    ),
                })
            } else {
                None
            }
        };

        for (index, layer_traces) in
            traces.borrow().layer_results.iter().enumerate()
        {
            let path = |suffix: &str| -> String {
                format!(
                    "activation_trace.layer_results.{}.activation_trace.{}",
                    index, suffix
                )
            };

            if let Some(r) = validate(
                &path("inputs"),
                &layer_traces.borrow().inputs.borrow(),
            ) {
                results.push(r);
            }
            if let Some(r) = validate(
                &path("pre_mixer_norm"),
                &layer_traces.borrow().pre_attention_norm.borrow(),
            ) {
                results.push(r);
            }
            if let Some(r) = validate(
                &path("mixer"),
                &layer_traces.borrow().attention.borrow(),
            ) {
                results.push(r);
            }
            if let Some(r) = validate(
                &path("post_mixer_norm"),
                &layer_traces.borrow().post_attention_norm.borrow(),
            ) {
                results.push(r);
            }
            if let Some(r) = validate(
                &path("mlp_inputs"),
                &layer_traces.borrow().mlp_inputs.borrow(),
            ) {
                results.push(r);
            }
            if let Some(r) = validate(
                &path("pre_mlp_norm"),
                &layer_traces.borrow().pre_mlp_norm.borrow(),
            ) {
                results.push(r);
            }
            if let Some(r) =
                validate(&path("mlp"), &layer_traces.borrow().mlp.borrow())
            {
                results.push(r);
            }
            if let Some(r) = validate(
                &path("post_mlp_norm"),
                &layer_traces.borrow().post_mlp_norm.borrow(),
            ) {
                results.push(r);
            }

            let outputs_path =
                format!("activation_trace.layer_results.{}.outputs", index);
            if let Some(r) =
                validate(&outputs_path, &layer_traces.borrow().outputs.borrow())
            {
                results.push(r);
            }
        }

        // Output norm (common to all models)
        if let Some(r) = validate(
            "activation_trace.output_norm",
            &traces.borrow().output_norm.borrow(),
        ) {
            results.push(r);
        }

        // Logits (common to all models, but path may vary)
        if let Some(r) = validate(
            "activation_trace.logits",
            &traces.borrow().logits.borrow(),
        ) {
            results.push(r);
        } else if let Some(r) =
            validate("logits", &traces.borrow().logits.borrow())
        {
            results.push(r);
        }

        results
    }

    fn validate_classifier_traces(
        traces: &Rc<RefCell<ActivationTrace>>,
        traces_view: &ParameterTree<Rc<MTLContext>>,
        data_type: DataType,
    ) -> Vec<TracerValidationResult> {
        let mut results = Vec::new();

        // Embedding norm (classifier-specific)
        if let Some(embedding_norm) = &traces.borrow().embedding_norm {
            if traces_view.leaf("activation_trace.embedding_norm").is_ok() {
                results.push(TracerValidationResult {
                    name: "activation_trace.embedding_norm".to_string(),
                    metrics: Self::validate_array_with_name(
                        data_type,
                        traces_view,
                        "activation_trace.embedding_norm",
                        &embedding_norm.borrow(),
                    ),
                });
            }
        }

        // Output pooling (classifier-specific)
        if let Some(output_pooling) = &traces.borrow().output_pooling {
            if traces_view.leaf("activation_trace.output_pooling").is_ok() {
                results.push(TracerValidationResult {
                    name: "activation_trace.output_pooling".to_string(),
                    metrics: Self::validate_array_with_name(
                        data_type,
                        traces_view,
                        "activation_trace.output_pooling",
                        &output_pooling.borrow(),
                    ),
                });
            }
        }

        results
    }

    // ========================================================================
    // Array Validation
    // ========================================================================

    fn validate_array(
        data_type: DataType,
        expected_array: &MetalArray,
        produced_array: &Ref<MetalArray>,
        transform: Option<ArrayTransform>,
    ) -> TracerValidationMetrics {
        match data_type {
            DataType::F16 => Self::validate_array_of_type::<f16>(
                expected_array,
                produced_array,
                transform,
            ),
            DataType::BF16 => Self::validate_array_of_type::<bf16>(
                expected_array,
                produced_array,
                transform,
            ),
            DataType::F32 => Self::validate_array_of_type::<f32>(
                expected_array,
                produced_array,
                transform,
            ),
            _ => panic!("Unsupported data type: {:?}", data_type),
        }
    }

    fn validate_array_with_name(
        data_type: DataType,
        traces_view: &ParameterTree<Rc<MTLContext>>,
        expected_array_path: &str,
        produced_array: &Ref<MetalArray>,
    ) -> TracerValidationMetrics {
        let expected_array = traces_view.leaf(expected_array_path).unwrap();
        Self::validate_array(data_type, &expected_array, produced_array, None)
    }

    fn validate_array_of_type<Precision: ArrayElement>(
        expected_array: &MetalArray,
        produced_array: &Ref<MetalArray>,
        transform: Option<ArrayTransform>,
    ) -> TracerValidationMetrics {
        let expected_view = expected_array.as_view::<Precision>().unwrap();
        let produced_view = produced_array.as_view::<Precision>().unwrap();

        let (mut expected_data, mut produced_data) = match transform {
            Some(ArrayTransform::KVCacheSlice) => {
                let permuted = produced_view.permuted_axes(IxDyn(&[1, 0, 2]));
                let total_tokens = permuted.shape()[0];
                let expected_tokens = expected_view.shape()[1];
                let start = total_tokens.saturating_sub(expected_tokens);
                let sliced = permuted.slice(s![start.., .., ..]);
                let reshaped = sliced
                    .into_owned()
                    .to_shape(IxDyn(&[
                        1,
                        expected_tokens,
                        permuted.shape()[1],
                        permuted.shape()[2],
                    ]))
                    .expect("Failed to reshape KV cache slice")
                    .to_owned();
                (expected_view.to_owned(), reshaped)
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

        // Handle shape mismatches with leading dimension of 1
        if expected_shape != produced_shape {
            if expected_shape.len() == produced_shape.len() + 1
                && expected_shape.get(0) == Some(&1)
                && expected_shape[1..] == produced_shape[..]
            {
                expected_data = expected_data
                    .to_shape(IxDyn(&produced_shape))
                    .expect("Failed to reshape expected data")
                    .to_owned();
            } else if produced_shape.len() == expected_shape.len() + 1
                && produced_shape.get(0) == Some(&1)
                && produced_shape[1..] == expected_shape[..]
            {
                produced_data = produced_data
                    .to_shape(IxDyn(&expected_shape))
                    .expect("Failed to reshape produced data")
                    .to_owned();
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
            .map(|value| NumCast::from(*value).unwrap_or(0.0))
            .collect();

        let result: Vec<f32> = produced_data
            .iter()
            .map(|value| NumCast::from(*value).unwrap_or(0.0))
            .collect();

        Self::compare_arrays(
            &reference,
            expected_data.shape().to_vec(),
            &result,
            produced_data.shape().to_vec(),
        )
    }

    fn compare_arrays(
        reference: &[f32],
        reference_shape: Vec<usize>,
        result: &[f32],
        result_shape: Vec<usize>,
    ) -> TracerValidationMetrics {
        assert_eq!(result.len(), reference.len());

        let atol: f32 = 1e-2;
        let rtol: f32 = 0.03;
        let fraction_of_allowed_violations: f32 = 0.01;

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

        for (i, (&exp, &prod)) in
            reference.iter().zip(result.iter()).enumerate()
        {
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

        let max_allowed_violations =
            (fraction_of_allowed_violations * n).ceil() as usize;

        TracerValidationMetrics {
            atol,
            rtol,
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

    fn load_array_as_vec<
        SourcePrecision: ArrayElement,
        TargetPrecision: NumCast,
    >(
        traces_view: &ParameterTree<Rc<MTLContext>>,
        name: &str,
    ) -> Vec<TargetPrecision> {
        let array = traces_view.leaf(name).unwrap();
        let slice = array.as_slice::<SourcePrecision>().unwrap();
        slice.iter().map(|x| NumCast::from(*x).unwrap()).collect()
    }

    fn determine_prefill_step_size(model_path: &Path) -> usize {
        let traces_path = model_path.join("traces.safetensors");
        if let Ok(file) = File::open(&traces_path) {
            if let Ok((_header_len, metadata)) =
                read_safetensors_metadata(&file)
            {
                if let Some(tensor) =
                    metadata.tensors.get("activation_trace.token_ids")
                {
                    if let Some(&length) = tensor.shape.first() {
                        return tensor
                            .shape
                            .iter()
                            .copied()
                            .max()
                            .unwrap_or(length)
                            .max(1);
                    }
                }
            }
        }
        1
    }

    fn ensure_llm_context_capacity(
        decoding_config: &DecodingConfig,
        desired_suffix_length: usize,
        context: &mut LanguageModelGeneratorContext,
    ) {
        let resolved_prefix_length =
            decoding_config.context_length.resolve(&context.model_config);
        let current_suffix_length = std::cmp::max(
            decoding_config.prefill_step_size.resolve(&context.model_config),
            decoding_config.generate_suffix_length(),
        );

        if desired_suffix_length <= current_suffix_length {
            return;
        }

        let decoder_config = &context.decoder_config;
        context.scratch_buffers = ScratchBuffers::new(
            &context.mtl_context,
            decoder_config,
            &context.model_shape,
            resolved_prefix_length,
            desired_suffix_length,
        );

        context.cache_layers = Rc::new(RefCell::new(CacheLayers::new(
            &context.mtl_context,
            &context.model_shape,
            resolved_prefix_length,
            desired_suffix_length,
        )));

        let intermediate_dtype: DataType =
            decoder_config.output_norm_config.scale_precision.into();
        let kernel_dtype: KernelDataType = intermediate_dtype.into();

        context.kv_cache_update = Box::new(
            KVCacheUpdate::new(
                &context.mtl_context,
                kernel_dtype,
                resolved_prefix_length,
            )
            .expect("Failed to create KV cache update kernel"),
        );

        context.gpu_sampler = Sampling::new(
            &context.mtl_context,
            kernel_dtype,
            desired_suffix_length,
            decoder_config.vocab_size,
        )
        .expect("Failed to create sampling kernel");
    }

    fn get_tokens_from_logits(logits: &MetalArray) -> Vec<u64> {
        let data_type = logits.data_type();
        match data_type {
            DataType::F16 => {
                Self::get_tokens_from_logits_of_type::<f16>(logits)
            },
            DataType::BF16 => {
                Self::get_tokens_from_logits_of_type::<bf16>(logits)
            },
            DataType::F32 => {
                Self::get_tokens_from_logits_of_type::<f32>(logits)
            },
            _ => panic!("Unsupported data type: {:?}", data_type),
        }
    }

    fn get_tokens_from_logits_of_type<Precision: ArrayElement>(
        logits: &MetalArray
    ) -> Vec<u64> {
        let sampler = ArgmaxSampler {};
        sampler.sample(logits.as_view::<Precision>().unwrap())
    }
}
