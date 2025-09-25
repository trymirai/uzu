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
            PrefillStepSize::Custom(1024),
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
            PrefillStepSize::Custom(1024),
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

        let token_ids = Self::load_array_as_vec::<i32, u64>(
            &traces_view,
            "activation_trace.token_ids".to_string(),
        );
        let token_positions = Self::load_array_as_vec::<i32, usize>(
            &traces_view,
            "activation_trace.token_positions".to_string(),
        );
        let mask: Vec<Vec<bool>> = Self::load_array_as_vec::<i8, i8>(
            &traces_view,
            "activation_trace.mask".to_string(),
        )
        .iter()
        .map(|x| *x != 0)
        .collect::<Vec<bool>>()
        .chunks(token_ids.len())
        .map(|chunk| chunk.to_vec())
        .collect();

        let external_bias_fn =
            |row: usize, col: usize| -> bool { !mask[row][col] };

        let mut state = ForwardPassState::new(
            self.generator_context.mtl_context.clone(),
            &self.generator_context.model_shape,
            &self.generator_context.scratch_buffers,
            self.generator_context.kv_cache.clone(),
            self.generator_context.shared_buffers.clone(),
            &token_ids,
            &token_positions,
            true,
            Some(&external_bias_fn),
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
        let mut results: Vec<TracerValidationResult> = Vec::new();

        let data_type =
            self.generator_context.model_shape.activation_data_type();
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
                results.push(TracerValidationResult {
                    name: expected_array_path.to_string(),
                    metrics,
                });
            };

        for (index, layer_traces) in
            traces.borrow().layer_results.iter().enumerate()
        {
            let path = |suffix: &str| -> String {
                format!(
                    "activation_trace.layer_results.{}.activation_trace.{}",
                    index.clone(),
                    suffix
                )
            };

            validate(
                path("inputs").as_str(),
                &layer_traces.borrow().inputs.borrow(),
                None,
            );
            validate(
                path("pre_attention_norm").as_str(),
                &layer_traces.borrow().pre_attention_norm.borrow(),
                None,
            );
            validate(
                path("attention").as_str(),
                &layer_traces.borrow().attention.borrow(),
                None,
            );
            if self.generator_context.executables.layers[index]
                .post_attention_norm
                .is_some()
            {
                validate(
                    path("post_attention_norm").as_str(),
                    &layer_traces.borrow().post_attention_norm.borrow(),
                    None,
                );
            }
            validate(
                path("mlp_inputs").as_str(),
                &layer_traces.borrow().mlp_inputs.borrow(),
                None,
            );
            validate(
                path("pre_mlp_norm").as_str(),
                &layer_traces.borrow().pre_mlp_norm.borrow(),
                None,
            );
            validate(
                path("mlp").as_str(),
                &layer_traces.borrow().mlp.borrow(),
                None,
            );
            if self.generator_context.executables.layers[index]
                .post_mlp_norm
                .is_some()
            {
                validate(
                    path("post_mlp_norm").as_str(),
                    &layer_traces.borrow().post_mlp_norm.borrow(),
                    None,
                );
            }
            validate(
                format!(
                    "activation_trace.layer_results.{}.outputs",
                    index.clone()
                )
                .as_str(),
                &layer_traces.borrow().outputs.borrow(),
                None,
            );
        }

        validate(
            "activation_trace.output_norm",
            &traces.borrow().output_norm.borrow(),
            None,
        );
        validate("logits", &traces.borrow().logits.borrow(), None);

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
            Self::get_tokens_from_logits(&*traces.borrow().logits.borrow());
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
        assert_eq!(expected_data.shape(), produced_data.shape());

        let reference = expected_data
            .iter()
            .map(|value| {
                let value_f32: f32 = NumCast::from(*value).unwrap();
                return value_f32;
            })
            .collect::<Vec<_>>();

        let result = produced_data
            .iter()
            .map(|value| {
                let value_f32: f32 = NumCast::from(*value).unwrap();
                return value_f32;
            })
            .collect::<Vec<_>>();

        let atol: f32 = 1e-2;
        let rtol: f32 = 0.03;
        let end_to_end_fraction_of_allowed_violations: f32 = 0.01;

        return Self::compare_arrays(
            &reference,
            expected_data.shape().to_vec(),
            &result,
            produced_data.shape().to_vec(),
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
