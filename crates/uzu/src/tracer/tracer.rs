use std::{
    cell::{Ref, RefCell},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    rc::Rc,
};

use half::{bf16, f16};
use ndarray::{IxDyn, s};
use num_traits::NumCast;

use crate::{
    Array, ArrayElement, DataType,
    backends::metal::{
        ForwardPassState, MTLContext, MetalArray,
        forward_pass::{
            ArrayId,
            encodable_with_state::{EncodableWithState, EncodingParameters},
            traces::DecoderActivationTrace,
        },
    },
    generator::{
        context::GeneratorContext,
        sampler::{ArgmaxSampler, LogitsSampler},
    },
    parameters::{ParameterLoader, ParameterTree},
    session::{
        config::{DecodingConfig, SpeculatorConfig},
        parameter::{ContextLength, PrefillStepSize, SamplingSeed},
    },
};

enum Transform {
    KVCacheSlice,
}

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
            return format!("Result contains NaN values");
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

pub struct TracerValidationResult {
    pub name: String,
    pub metrics: TracerValidationMetrics,
}

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

pub struct Tracer {
    model_path: PathBuf,
    generator_context: GeneratorContext,
}

impl Tracer {
    pub fn new(model_path: &Path) -> Self {
        let decoding_config = DecodingConfig::new(
            PrefillStepSize::Custom(1),
            ContextLength::default(),
            SpeculatorConfig::default(),
            SamplingSeed::default(),
            false,
        );
        let generator_context =
            GeneratorContext::new(model_path, &decoding_config).unwrap();

        Self {
            model_path: model_path.to_path_buf(),
            generator_context,
        }
    }

    pub fn new_with_config(
        model_path: &Path,
        sampling_seed: SamplingSeed,
        max_prefix_length: ContextLength,
    ) -> Self {
        let decoding_config = DecodingConfig::new(
            PrefillStepSize::Custom(1),
            max_prefix_length,
            SpeculatorConfig::default(),
            sampling_seed,
            false,
        );
        let generator_context =
            GeneratorContext::new(model_path, &decoding_config).unwrap();

        Self {
            model_path: model_path.to_path_buf(),
            generator_context,
        }
    }

    pub fn run(&self) -> TracerValidationResults {
        let traces_path = self.model_path.join("traces.safetensors");
        if !traces_path.exists() {
            panic!("Traces file not found at {:?}", traces_path);
        }
        let traces_file =
            File::open(&traces_path).expect("Failed to open traces file");
        let traces_loader = ParameterLoader::new(
            &traces_file,
            &self.generator_context.mtl_context,
        )
        .expect("Failed to create ParameterLoader for traces");
        let traces_view = traces_loader.tree();

        let mut token_ids = Self::load_array_as_vec::<i32, u64>(
            &traces_view,
            "activation_trace.token_ids".to_string(),
        );
        let mut token_positions = Self::load_array_as_vec::<i32, usize>(
            &traces_view,
            "activation_trace.token_positions".to_string(),
        );

        let max_suffix_length =
            self.generator_context.kv_cache.borrow().max_suffix_length();
        token_ids.truncate(max_suffix_length);
        token_positions.truncate(max_suffix_length);

        let mut state = ForwardPassState::new(
            self.generator_context.mtl_context.clone(),
            &self.generator_context.model_config.decoder_config,
            &self.generator_context.model_shape,
            &self.generator_context.scratch_buffers,
            self.generator_context.kv_cache.clone(),
            self.generator_context.shared_buffers.clone(),
            &token_ids,
            &token_positions,
            true,
            None,
        );

        let root_command_buffer = self
            .generator_context
            .command_buffer
            .root_command_buffer()
            .to_owned();
        self.generator_context.executables.encode(
            &mut state,
            &self.generator_context.command_buffer,
            &EncodingParameters::new(false, false, true),
        );
        self.generator_context.command_buffer.commit();
        root_command_buffer.wait_until_completed();

        let traces = state.traces.clone().unwrap();
        let results =
            self.validate_traces(token_ids.len(), &state, &traces_view, traces);
        return results;
    }
}

impl Tracer {
    fn validate_traces(
        &self,
        suffix_length: usize,
        state: &ForwardPassState,
        traces_view: &ParameterTree<Rc<MTLContext>>,
        traces: Rc<RefCell<DecoderActivationTrace>>,
    ) -> TracerValidationResults {
        let mut base_results: Vec<TracerValidationResult> = Vec::new();
        let mut moe_results_all: Vec<TracerValidationResult> = Vec::new();

        let data_type =
            self.generator_context.model_shape.activation_data_type();

        if std::env::var_os("UZU_DEBUG_MOE_STATE").is_some() {
            self.debug_moe_state(state, data_type);
        }
        let mut validate =
            |expected_array_path: &str,
             produced_array: &Ref<MetalArray>,
             produced_transform: Option<Transform>| {
                let metrics = Self::validate_array_with_name(
                    data_type,
                    traces_view,
                    expected_array_path.to_string(),
                    produced_array,
                    produced_transform,
                );
                base_results.push(TracerValidationResult {
                    name: expected_array_path.to_string(),
                    metrics,
                });
            };

        let traces_ref = traces.borrow();
        for (index, layer_traces) in traces_ref.layer_results.iter().enumerate()
        {
            let layer_trace_ref = layer_traces.borrow();
            let path = |suffix: &str| -> String {
                format!(
                    "activation_trace.layer_results.{}.activation_trace.{}",
                    index.clone(),
                    suffix
                )
            };

            validate(
                path("inputs").as_str(),
                &layer_trace_ref.inputs.borrow(),
                None,
            );
            validate(
                path("pre_attention_norm").as_str(),
                &layer_trace_ref.pre_attention_norm.borrow(),
                None,
            );
            validate(
                path("attention").as_str(),
                &layer_trace_ref.attention.borrow(),
                None,
            );
            if self.generator_context.executables.layers[index]
                .post_attention_norm
                .is_some()
            {
                validate(
                    path("post_attention_norm").as_str(),
                    &layer_trace_ref.post_attention_norm.borrow(),
                    None,
                );
            }
            validate(
                path("mlp_inputs").as_str(),
                &layer_trace_ref.mlp_inputs.borrow(),
                None,
            );
            validate(
                path("pre_mlp_norm").as_str(),
                &layer_trace_ref.pre_mlp_norm.borrow(),
                None,
            );
            validate(path("mlp").as_str(), &layer_trace_ref.mlp.borrow(), None);
            if self.generator_context.executables.layers[index]
                .post_mlp_norm
                .is_some()
            {
                validate(
                    path("post_mlp_norm").as_str(),
                    &layer_trace_ref.post_mlp_norm.borrow(),
                    None,
                );
            }
            validate(
                format!(
                    "activation_trace.layer_results.{}.outputs",
                    index.clone()
                )
                .as_str(),
                &layer_trace_ref.outputs.borrow(),
                None,
            );

            if let Some(moe_trace) = layer_trace_ref.moe.as_ref() {
                let moe_base =
                    format!("activation_trace.layer_results.{}.moe", index);

                Self::maybe_validate_moe_vec_i32(
                    &mut moe_results_all,
                    traces_view,
                    format!("{}.topk_ids", moe_base),
                    &moe_trace.topk_ids,
                );
                Self::maybe_validate_moe_vec_u32(
                    &mut moe_results_all,
                    traces_view,
                    format!("{}.counts", moe_base),
                    &moe_trace.counts,
                );
                Self::maybe_validate_moe_vec_u32(
                    &mut moe_results_all,
                    traces_view,
                    format!("{}.offsets", moe_base),
                    &moe_trace.offsets,
                );
                Self::maybe_validate_moe_vec_u32(
                    &mut moe_results_all,
                    traces_view,
                    format!("{}.sumk", moe_base),
                    &moe_trace.sumk,
                );
                Self::maybe_validate_moe_vec_u32(
                    &mut moe_results_all,
                    traces_view,
                    format!("{}.bucketed_ids", moe_base),
                    &moe_trace.bucketed_ids,
                );
                Self::maybe_validate_moe_vec_i32(
                    &mut moe_results_all,
                    traces_view,
                    format!("{}.tok2row", moe_base),
                    &moe_trace.tok2row,
                );
                Self::maybe_validate_moe_vec_f32(
                    &mut moe_results_all,
                    traces_view,
                    format!("{}.topk_probs", moe_base),
                    &moe_trace.topk_probs,
                );
                Self::maybe_validate_moe_vec_f32(
                    &mut moe_results_all,
                    traces_view,
                    format!("{}.bucketed_probs", moe_base),
                    &moe_trace.bucketed_probs,
                );
                Self::maybe_validate_moe_vec_f32(
                    &mut moe_results_all,
                    traces_view,
                    format!("{}.y_partial", moe_base),
                    &moe_trace.y_partial,
                );
            }
        }

        validate(
            "activation_trace.output_norm",
            &traces_ref.output_norm.borrow(),
            None,
        );
        validate("logits", &traces_ref.logits.borrow(), None);

        for index in 0..state.kv_cache.borrow().data.len() {
            let arrays =
                state.arrays(&[ArrayId::Keys(index), ArrayId::Values(index)]);

            let keys = arrays[0].borrow();
            validate(
                format!("updated_kv_cache.{}.keys", index.clone()).as_str(),
                &keys,
                Some(Transform::KVCacheSlice),
            );

            let values = arrays[1].borrow();
            validate(
                format!("updated_kv_cache.{}.values", index.clone()).as_str(),
                &values,
                Some(Transform::KVCacheSlice),
            );
        }

        let expected_tokens =
            Self::get_tokens_from_logits(&traces_view.leaf("logits").unwrap());
        let produced_tokens =
            Self::get_tokens_from_logits(&*traces_ref.logits.borrow());
        let tokens_violation_indices: Vec<usize> = expected_tokens
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

        let mut results = base_results;
        results.extend(moe_results_all);

        if let Some(dump_dir_os) = std::env::var_os("UZU_DEBUG_MOE_DUMP") {
            let dump_path = PathBuf::from(dump_dir_os);
            if let Err(error) = Self::dump_moe_traces(&dump_path, &traces_ref) {
                eprintln!(
                    "[DebugMoeState] failed to dump MoE traces to {:?}: {}",
                    dump_path, error
                );
            }
        }

        drop(traces_ref);

        return TracerValidationResults {
            suffix_length,
            results,
            tokens_violation_indices,
        };
    }

    fn validate_array_with_name(
        data_type: DataType,
        traces_view: &ParameterTree<Rc<MTLContext>>,
        expected_array_path: String,
        produced_array: &Ref<MetalArray>,
        produced_transform: Option<Transform>,
    ) -> TracerValidationMetrics {
        let expected_array =
            traces_view.leaf(expected_array_path.as_str()).unwrap();

        if std::env::var_os("UZU_DEBUG_MOE_MLP").is_some()
            && expected_array_path.contains("activation_trace.mlp")
        {
            Self::debug_mlp_sample(
                data_type,
                expected_array_path.as_str(),
                &expected_array,
                produced_array,
            );
        }
        return Self::validate_array(
            data_type,
            &expected_array,
            produced_array,
            produced_transform,
        );
    }

    fn validate_array(
        data_type: DataType,
        expected_array: &MetalArray,
        produced_array: &Ref<MetalArray>,
        produced_transform: Option<Transform>,
    ) -> TracerValidationMetrics {
        match data_type {
            DataType::F16 => {
                return Self::validate_array_of_type::<f16>(
                    expected_array,
                    produced_array,
                    produced_transform,
                );
            },
            DataType::BF16 => {
                return Self::validate_array_of_type::<bf16>(
                    expected_array,
                    produced_array,
                    produced_transform,
                );
            },
            DataType::F32 => {
                return Self::validate_array_of_type::<f32>(
                    expected_array,
                    produced_array,
                    produced_transform,
                );
            },
            _ => panic!("Unsupported data type: {:?}", data_type),
        }
    }

    fn validate_array_of_type<Precision: ArrayElement>(
        expected_array: &MetalArray,
        produced_array: &Ref<MetalArray>,
        produced_transform: Option<Transform>,
    ) -> TracerValidationMetrics {
        let expected_view = expected_array.as_view::<Precision>().unwrap();
        let produced_view = produced_array.as_view::<Precision>().unwrap();

        let (expected_data, produced_data) = match produced_transform {
            Some(transform) => match transform {
                Transform::KVCacheSlice => {
                    let transformed_data = produced_view
                        .permuted_axes(IxDyn(&[1, 0, 2]))
                        .slice(s![0..expected_view.shape()[0], .., ..])
                        .into_dyn()
                        .to_owned();
                    (expected_view.to_owned(), transformed_data)
                },
            },
            None => (expected_view.to_owned(), produced_view.to_owned()),
        };

        let produced_len = produced_data.len();
        let expected_len = expected_data.len();
        assert!(
            produced_len <= expected_len,
            "Produced array length {} exceeds reference length {}",
            produced_len,
            expected_len
        );

        let reference = expected_data
            .iter()
            .take(produced_len)
            .map(|value| {
                let value_f32: f32 = NumCast::from(*value).unwrap();
                value_f32
            })
            .collect::<Vec<_>>();

        let result = produced_data
            .iter()
            .take(produced_len)
            .map(|value| {
                let value_f32: f32 = NumCast::from(*value).unwrap();
                value_f32
            })
            .collect::<Vec<_>>();

        if std::env::var("UZU_DEBUG_ATTENTION").is_ok() {
            let preview = |label: &str, values: &Vec<f32>| {
                let slice: Vec<_> = values.iter().take(8).copied().collect();
                eprintln!("[TracerDebug] {} preview {:?}", label, slice);
            };
            preview("reference", &reference);
            preview("result", &result);
        }

        let atol: f32 = 1e-2;
        let rtol: f32 = 0.03;
        let end_to_end_fraction_of_allowed_violations: f32 = 0.01;

        let produced_shape = produced_data.shape().to_vec();

        return Self::compare_arrays(
            &reference,
            produced_shape.clone(),
            &result,
            produced_shape,
            atol,
            rtol,
            end_to_end_fraction_of_allowed_violations,
        );
    }

    fn compare_arrays(
        reference: &Vec<f32>,
        reference_shape: Vec<usize>,
        result: &Vec<f32>,
        result_shape: Vec<usize>,
        atol: f32,
        rtol: f32,
        fraction_of_allowed_violations: f32,
    ) -> TracerValidationMetrics {
        assert_eq!(result.len(), reference.len());

        let absdiff: Vec<f32> = result
            .iter()
            .zip(reference.iter())
            .map(|(result_value, reference_value)| {
                (result_value - reference_value).abs()
            })
            .collect();

        let allowed_diff: Vec<f32> = reference
            .iter()
            .map(|reference_value| atol + rtol * reference_value.abs())
            .collect();

        let violations: Vec<f32> = absdiff
            .iter()
            .zip(allowed_diff.iter())
            .map(|(absdiff_value, allowed_diff_value)| {
                (absdiff_value - allowed_diff_value).max(0.0)
            })
            .collect();

        let num_violations =
            violations.iter().filter(|&&value| value > 0.0).count();

        let max_allowed_violations = (reference.len() as f32
            * fraction_of_allowed_violations)
            .ceil() as usize;

        let err_rel: Vec<f32> = absdiff
            .iter()
            .zip(reference.iter())
            .map(|(absdiff_value, reference_value)| {
                absdiff_value / (reference_value.abs() + 1e-10)
            })
            .collect();

        let (max_err_idx, _) = violations
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let max_err = absdiff[max_err_idx];
        let max_err_rel = err_rel[max_err_idx];
        let max_err_reference_value = reference[max_err_idx];

        let rms_diff = {
            let sum_squares: f32 = absdiff.iter().map(|x| x * x).sum();
            (sum_squares / absdiff.len() as f32).sqrt()
        };

        let rms_result = {
            let sum_squares: f32 = result.iter().map(|x| x * x).sum();
            (sum_squares / result.len() as f32).sqrt()
        };

        let rms_reference = {
            let sum_squares: f32 = reference.iter().map(|x| x * x).sum();
            (sum_squares / reference.len() as f32).sqrt()
        };

        let rel_rms_reference = rms_diff / (rms_reference + 1e-10);

        let diff_max = absdiff.iter().cloned().reduce(|a, b| a.max(b)).unwrap();
        let diff_avg =
            absdiff.iter().cloned().sum::<f32>() / absdiff.len() as f32;

        let result_nan = result.iter().any(|x| x.is_nan());

        return TracerValidationMetrics {
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
        };
    }

    fn maybe_validate_moe_vec_i32(
        results: &mut Vec<TracerValidationResult>,
        traces_view: &ParameterTree<Rc<MTLContext>>,
        path: String,
        data: &[i32],
    ) {
        if let Ok(expected_array) = traces_view.leaf(path.as_str()) {
            if let Ok(expected_slice) = expected_array.as_slice::<i32>() {
                let expected_shape = expected_array.shape().to_vec();
                let reference: Vec<f32> =
                    expected_slice.iter().map(|&v| v as f32).collect();
                let result: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                let len = reference.len().min(result.len());
                if len == 0 {
                    return;
                }
                let reference_vec = reference[..len].to_vec();
                let result_vec = result[..len].to_vec();
                let final_shape = if len == reference.len() {
                    expected_shape.clone()
                } else {
                    vec![len]
                };
                let metrics = Self::compare_arrays(
                    &reference_vec,
                    final_shape.clone(),
                    &result_vec,
                    final_shape,
                    0.0,
                    0.0,
                    0.0,
                );
                results.push(TracerValidationResult {
                    name: path,
                    metrics,
                });
            }
        }
    }

    fn maybe_validate_moe_vec_u32(
        results: &mut Vec<TracerValidationResult>,
        traces_view: &ParameterTree<Rc<MTLContext>>,
        path: String,
        data: &[u32],
    ) {
        if let Ok(expected_array) = traces_view.leaf(path.as_str()) {
            if let Ok(expected_slice) = expected_array.as_slice::<u32>() {
                let expected_shape = expected_array.shape().to_vec();
                let reference: Vec<f32> =
                    expected_slice.iter().map(|&v| v as f32).collect();
                let result: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                let len = reference.len().min(result.len());
                if len == 0 {
                    return;
                }
                let reference_vec = reference[..len].to_vec();
                let result_vec = result[..len].to_vec();
                let final_shape = if len == reference.len() {
                    expected_shape.clone()
                } else {
                    vec![len]
                };
                let metrics = Self::compare_arrays(
                    &reference_vec,
                    final_shape.clone(),
                    &result_vec,
                    final_shape,
                    0.0,
                    0.0,
                    0.0,
                );
                results.push(TracerValidationResult {
                    name: path,
                    metrics,
                });
            }
        }
    }

    fn maybe_validate_moe_vec_f32(
        results: &mut Vec<TracerValidationResult>,
        traces_view: &ParameterTree<Rc<MTLContext>>,
        path: String,
        data: &[f32],
    ) {
        if let Ok(expected_array) = traces_view.leaf(path.as_str()) {
            if let Some(reference) =
                Self::metal_array_to_vec_f32(&expected_array)
            {
                let expected_shape = expected_array.shape().to_vec();
                let len = reference.len().min(data.len());
                if len == 0 {
                    return;
                }
                let result = data[..len].to_vec();
                let reference_trimmed = reference[..len].to_vec();
                let final_shape = if len == reference.len() {
                    expected_shape.clone()
                } else {
                    vec![len]
                };
                let metrics = Self::compare_arrays(
                    &reference_trimmed,
                    final_shape.clone(),
                    &result,
                    final_shape,
                    1e-2,
                    0.03,
                    0.01,
                );
                results.push(TracerValidationResult {
                    name: path,
                    metrics,
                });
            }
        }
    }

    fn metal_array_to_vec_f32(array: &MetalArray) -> Option<Vec<f32>> {
        match array.data_type() {
            DataType::BF16 => array
                .as_slice::<bf16>()
                .ok()
                .map(|slice| slice.iter().map(|v| v.to_f32()).collect()),
            DataType::F16 => array
                .as_slice::<f16>()
                .ok()
                .map(|slice| slice.iter().map(|v| v.to_f32()).collect()),
            DataType::F32 => {
                array.as_slice::<f32>().ok().map(|slice| slice.to_vec())
            },
            _ => None,
        }
    }

    fn dump_moe_traces(
        dump_path: &Path,
        traces: &DecoderActivationTrace,
    ) -> std::io::Result<()> {
        fs::create_dir_all(dump_path)?;

        for (layer_index, layer_trace_rc) in
            traces.layer_results.iter().enumerate()
        {
            let layer_trace = layer_trace_rc.borrow();
            if let Some(moe) = &layer_trace.moe {
                let layer_dir = dump_path.join(format!("layer_{layer_index}"));
                fs::create_dir_all(&layer_dir)?;

                Self::write_moe_meta(&layer_dir, moe)?;
                Self::write_vec_to_file(
                    layer_dir.join("topk_ids.txt"),
                    &moe.topk_ids,
                )?;
                Self::write_vec_to_file(
                    layer_dir.join("counts.txt"),
                    &moe.counts,
                )?;
                Self::write_vec_to_file(
                    layer_dir.join("offsets.txt"),
                    &moe.offsets,
                )?;
                Self::write_vec_to_file(layer_dir.join("sumk.txt"), &moe.sumk)?;
                Self::write_vec_to_file(
                    layer_dir.join("bucketed_ids.txt"),
                    &moe.bucketed_ids,
                )?;
                Self::write_vec_to_file(
                    layer_dir.join("tok2row.txt"),
                    &moe.tok2row,
                )?;
                Self::write_vec_to_file(
                    layer_dir.join("topk_probs.txt"),
                    &moe.topk_probs,
                )?;
                Self::write_vec_to_file(
                    layer_dir.join("bucketed_probs.txt"),
                    &moe.bucketed_probs,
                )?;
                Self::write_vec_to_file(
                    layer_dir.join("y_partial.txt"),
                    &moe.y_partial,
                )?;
            }
        }

        Ok(())
    }

    fn write_moe_meta(
        layer_dir: &Path,
        moe: &crate::backends::metal::forward_pass::traces::MoeActivationTrace,
    ) -> std::io::Result<()> {
        let meta_path = layer_dir.join("meta.txt");
        let mut file = File::create(meta_path)?;
        writeln!(file, "suffix_length={}", moe.suffix_length)?;
        writeln!(file, "mixture_size={}", moe.mixture_size)?;
        writeln!(file, "k={}", moe.k)?;
        writeln!(file, "model_dim={}", moe.model_dim)?;
        Ok(())
    }

    fn write_vec_to_file<T: std::fmt::Display>(
        path: impl AsRef<Path>,
        data: &[T],
    ) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        for (index, value) in data.iter().enumerate() {
            if index > 0 {
                if index % 16 == 0 {
                    file.write_all(b"\n")?;
                } else {
                    file.write_all(b" ")?;
                }
            }
            write!(file, "{}", value)?;
        }
        file.write_all(b"\n")?;
        Ok(())
    }

    fn debug_mlp_sample(
        data_type: DataType,
        array_path: &str,
        expected_array: &MetalArray,
        produced_array: &Ref<MetalArray>,
    ) {
        let len =
            expected_array.num_elements().min(produced_array.num_elements());
        if len == 0 {
            return;
        }

        let inspect = [0usize, 1, 2, 1167];

        match data_type {
            DataType::BF16 => {
                if let (Ok(ref_slice), Ok(prod_slice)) = (
                    expected_array.as_slice::<bf16>(),
                    produced_array.as_slice::<bf16>(),
                ) {
                    Self::print_samples(array_path, len, &inspect, |idx| {
                        (prod_slice[idx].to_f32(), ref_slice[idx].to_f32())
                    });
                }
            },
            DataType::F16 => {
                if let (Ok(ref_slice), Ok(prod_slice)) = (
                    expected_array.as_slice::<f16>(),
                    produced_array.as_slice::<f16>(),
                ) {
                    Self::print_samples(array_path, len, &inspect, |idx| {
                        (prod_slice[idx].to_f32(), ref_slice[idx].to_f32())
                    });
                }
            },
            DataType::F32 => {
                if let (Ok(ref_slice), Ok(prod_slice)) = (
                    expected_array.as_slice::<f32>(),
                    produced_array.as_slice::<f32>(),
                ) {
                    Self::print_samples(array_path, len, &inspect, |idx| {
                        (prod_slice[idx], ref_slice[idx])
                    });
                }
            },
            _ => {},
        }
    }

    fn print_samples<F>(
        array_path: &str,
        len: usize,
        inspect: &[usize],
        mut value_fn: F,
    ) where
        F: FnMut(usize) -> (f32, f32),
    {
        let _ = (array_path, len, inspect, &mut value_fn);
    }

    fn debug_moe_state(
        &self,
        state: &ForwardPassState,
        data_type: DataType,
    ) {
        let _ = (state, data_type); // logging disabled
    }

    fn load_array_as_vec<
        SourcePrecision: ArrayElement,
        TargetPrecision: NumCast,
    >(
        traces_view: &ParameterTree<Rc<MTLContext>>,
        name: String,
    ) -> Vec<TargetPrecision> {
        let array = traces_view.leaf(name.as_str()).unwrap();
        let slice = array.as_slice::<SourcePrecision>().unwrap();
        let vec: Vec<TargetPrecision> = slice
            .iter()
            .map(|x| {
                let casted_value: TargetPrecision = NumCast::from(*x).unwrap();
                return casted_value;
            })
            .collect();
        return vec;
    }

    fn get_tokens_from_logits(logits: &MetalArray) -> Vec<u64> {
        let data_type = logits.data_type();
        match data_type {
            DataType::F16 => {
                return Self::get_tokens_from_logits_of_type::<f16>(logits);
            },
            DataType::BF16 => {
                return Self::get_tokens_from_logits_of_type::<bf16>(logits);
            },
            DataType::F32 => {
                return Self::get_tokens_from_logits_of_type::<f32>(logits);
            },
            _ => panic!("Unsupported data type: {:?}", data_type),
        }
    }

    fn get_tokens_from_logits_of_type<Precision: ArrayElement>(
        logits: &MetalArray
    ) -> Vec<u64> {
        let sampler = ArgmaxSampler {};
        let tokens = sampler.sample(logits.as_view::<Precision>().unwrap());
        return tokens;
    }
}
