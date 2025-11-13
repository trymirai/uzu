use std::{
    cell::{Ref, RefCell},
    path::{Path, PathBuf},
    rc::Rc,
};

use half::{bf16, f16};
use ndarray::IxDyn;
use num_traits::NumCast;

use crate::{
    Array, ArrayElement, DataType,
    backends::metal::{MTLContext, MetalArray},
    classifier::{Classifier, ClassifierActivationTrace},
    parameters::{ParameterLoader, ParameterTree},
    tracer::tracer::{
        TracerValidationMetrics, TracerValidationResult,
        TracerValidationResults,
    },
};

pub struct ClassifierTraceValidator {
    model_path: PathBuf,
    classifier: Classifier,
}

impl ClassifierTraceValidator {
    pub fn new(model_path: &Path) -> Self {
        let classifier =
            Classifier::new(model_path).expect("Failed to create Classifier");

        Self {
            model_path: model_path.to_path_buf(),
            classifier,
        }
    }

    pub fn run(&mut self) -> TracerValidationResults {
        let traces_path = self.model_path.join("traces.safetensors");
        if !traces_path.exists() {
            panic!("Traces file not found at {:?}", traces_path);
        }

        let traces_file = std::fs::File::open(&traces_path)
            .expect("Failed to open traces file");

        // Clone mtl_context to avoid borrow issues
        let mtl_context_clone = self.classifier.context.mtl_context.clone();

        let traces_loader =
            ParameterLoader::new(&traces_file, &mtl_context_clone)
                .expect("Failed to create ParameterLoader for traces");
        let traces_view = traces_loader.tree();

        // Load token IDs and positions from traces
        let token_ids = Self::load_array_as_vec::<i32, u64>(
            &traces_view,
            "activation_trace.token_ids".to_string(),
        );
        let token_positions = Self::load_array_as_vec::<i32, usize>(
            &traces_view,
            "activation_trace.token_positions".to_string(),
        );

        let suffix_length = token_ids.len();

        // Run full forward pass with traces enabled using the classifier API
        // This runs the entire pipeline with all buffers pre-allocated
        let (_logits, traces) = self
            .classifier
            .forward_pass_with_traces(&token_ids, &token_positions)
            .expect("Failed to run forward pass with traces");

        // Validate traces
        let results = self.validate_traces(suffix_length, &traces_view, traces);

        results
    }

    fn validate_traces(
        &self,
        suffix_length: usize,
        traces_view: &ParameterTree<Rc<MTLContext>>,
        traces: Rc<RefCell<ClassifierActivationTrace>>,
    ) -> TracerValidationResults {
        let mut results: Vec<TracerValidationResult> = Vec::new();

        let data_type =
            self.classifier.context.model_shape.activation_data_type();

        let mut validate =
            |expected_array_path: &str, produced_array: &Ref<MetalArray>| {
                let metrics = Self::validate_array_with_name(
                    data_type,
                    traces_view,
                    expected_array_path.to_string(),
                    produced_array,
                );
                results.push(TracerValidationResult {
                    name: expected_array_path.to_string(),
                    metrics,
                });
            };

        // Note: embedding_norm_output is not exported by lalamo, so we skip it

        // Validate each layer
        for (index, layer_traces) in
            traces.borrow().layer_results.iter().enumerate()
        {
            let path = |suffix: &str| -> String {
                format!(
                    "activation_trace.layer_results.{}.activation_trace.{}",
                    index, suffix
                )
            };

            validate(
                path("inputs").as_str(),
                &layer_traces.borrow().inputs.borrow(),
            );

            // Check if pre_attention_norm exists (first layer might not have it)
            if traces_view.leaf(path("pre_attention_norm").as_str()).is_ok() {
                validate(
                    path("pre_attention_norm").as_str(),
                    &layer_traces.borrow().pre_attention_norm.borrow(),
                );
            }

            validate(
                path("attention").as_str(),
                &layer_traces.borrow().attention.borrow(),
            );
            validate(
                path("mlp_inputs").as_str(),
                &layer_traces.borrow().mlp_inputs.borrow(),
            );
            validate(
                path("pre_mlp_norm").as_str(),
                &layer_traces.borrow().pre_mlp_norm.borrow(),
            );
            validate(path("mlp").as_str(), &layer_traces.borrow().mlp.borrow());
            validate(
                format!("activation_trace.layer_results.{}.outputs", index)
                    .as_str(),
                &layer_traces.borrow().outputs.borrow(),
            );
        }

        // Validate output_norm
        validate(
            "activation_trace.output_norm",
            &traces.borrow().output_norm.borrow(),
        );

        // Validate output_pooling
        validate(
            "activation_trace.output_pooling",
            &traces.borrow().output_pooling.borrow(),
        );

        // Validate logits
        validate("activation_trace.logits", &traces.borrow().logits.borrow());

        TracerValidationResults {
            suffix_length,
            results,
            tokens_violation_indices: Vec::new(), // Not applicable for classification
        }
    }

    fn validate_array_with_name(
        data_type: DataType,
        traces_view: &ParameterTree<Rc<MTLContext>>,
        expected_array_path: String,
        produced_array: &Ref<MetalArray>,
    ) -> TracerValidationMetrics {
        let expected_array =
            traces_view.leaf(expected_array_path.as_str()).unwrap();
        Self::validate_array(data_type, &expected_array, produced_array)
    }

    fn validate_array(
        data_type: DataType,
        expected_array: &MetalArray,
        produced_array: &Ref<MetalArray>,
    ) -> TracerValidationMetrics {
        match data_type {
            DataType::F16 => Self::validate_array_of_type::<f16>(
                expected_array,
                produced_array,
            ),
            DataType::BF16 => Self::validate_array_of_type::<bf16>(
                expected_array,
                produced_array,
            ),
            DataType::F32 => Self::validate_array_of_type::<f32>(
                expected_array,
                produced_array,
            ),
            _ => panic!("Unsupported data type: {:?}", data_type),
        }
    }

    fn validate_array_of_type<Precision: ArrayElement>(
        expected_array: &MetalArray,
        produced_array: &Ref<MetalArray>,
    ) -> TracerValidationMetrics {
        let expected_view = expected_array.as_view::<Precision>().unwrap();
        let produced_view = produced_array.as_view::<Precision>().unwrap();

        let expected_data = expected_view.to_owned();
        let produced_data = produced_view.to_owned();

        let expected_shape = expected_data.shape().to_vec();
        let produced_shape = produced_data.shape().to_vec();

        // Handle shape mismatches (e.g., batch dimension)
        let (expected_data, produced_data) = if expected_shape != produced_shape
        {
            if expected_shape.len() == produced_shape.len() + 1
                && expected_shape.get(0) == Some(&1)
                && expected_shape[1..] == produced_shape[..]
            {
                // Expected has batch dimension of 1, produced doesn't
                let reshaped = produced_data
                    .to_shape(IxDyn(&expected_shape))
                    .expect("Failed to reshape")
                    .to_owned();
                (expected_data, reshaped)
            } else if produced_shape.len() == expected_shape.len() + 1
                && produced_shape.get(0) == Some(&1)
                && produced_shape[1..] == expected_shape[..]
            {
                // Produced has batch dimension of 1, expected doesn't
                let reshaped = expected_data
                    .to_shape(IxDyn(&produced_shape))
                    .expect("Failed to reshape")
                    .to_owned();
                (reshaped, produced_data)
            } else {
                panic!(
                    "Shape mismatch: expected {:?}, produced {:?}",
                    expected_shape, produced_shape
                );
            }
        } else {
            (expected_data, produced_data)
        };

        // Compute validation metrics
        let atol = 0.05;
        let rtol = 0.01;
        let fraction_of_allowed_violations = 0.03;

        let expected_flat: Vec<f32> =
            expected_data.iter().map(|x| x.to_f32().unwrap_or(0.0)).collect();
        let produced_flat: Vec<f32> =
            produced_data.iter().map(|x| x.to_f32().unwrap_or(0.0)).collect();

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
            expected_flat.iter().zip(produced_flat.iter()).enumerate()
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

        let n = expected_flat.len() as f32;
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
            (fraction_of_allowed_violations * n) as usize;

        TracerValidationMetrics {
            atol,
            rtol,
            fraction_of_allowed_violations,
            reference_shape: expected_shape,
            result_shape: produced_shape,
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
                casted_value
            })
            .collect();
        vec
    }
}
