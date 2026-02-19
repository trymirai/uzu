use std::collections::HashMap;

use metal::{MTLBuffer, MTLComputeCommandEncoder};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{
    dispatch_descriptor::{DispatchDescriptor, OutputSource},
    pipeline_configuration::PipelineConfiguration,
};
use crate::{
    DataType,
    backends::{
        common::kernel::{
            MatmulGemvShape0Kernel, MatmulGemvShape1Kernel, MatmulGemvShape2Kernel, MatmulGemvShape3Kernel,
            MatmulGemvShape4Kernel, MatmulGemvShape5Kernel, MatmulGemvShape6Kernel,
        },
        metal::{
            MetalContext, MetalError,
            kernel::{
                dsl::{
                    MatmulGemvShape0MetalKernel, MatmulGemvShape1MetalKernel, MatmulGemvShape2MetalKernel,
                    MatmulGemvShape3MetalKernel, MatmulGemvShape4MetalKernel, MatmulGemvShape5MetalKernel,
                    MatmulGemvShape6MetalKernel,
                },
                matmul::common::MatmulArguments,
            },
        },
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GemvShape {
    Shape0,
    Shape1,
    Shape2,
    Shape3,
    Shape4,
    Shape5,
    Shape6,
}

impl GemvShape {
    fn from_pipeline_configuration(config: &PipelineConfiguration) -> Option<Self> {
        if config.transpose_matrix || config.batch_pack != 1 || config.non_contiguous_batch {
            return None;
        }

        let shape = (
            config.threadgroup_rows,
            config.threadgroup_cols,
            config.threads_per_simdgroup_row,
            config.threads_per_simdgroup_col,
            config.elements_per_thread_row,
            config.elements_per_thread_col,
        );

        match shape {
            (1, 8, 1, 32, 4, 4) => Some(Self::Shape0),
            (1, 8, 1, 32, 1, 4) => Some(Self::Shape1),
            (1, 1, 8, 4, 4, 4) => Some(Self::Shape2),
            (1, 1, 8, 4, 1, 4) => Some(Self::Shape3),
            (4, 1, 1, 32, 1, 4) => Some(Self::Shape4),
            (4, 1, 1, 32, 4, 4) => Some(Self::Shape5),
            (8, 1, 1, 32, 4, 4) => Some(Self::Shape6),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    shape: GemvShape,
    apply_output_scale_and_accumulate: bool,
}

enum DslPipeline {
    Shape0(MatmulGemvShape0MetalKernel),
    Shape1(MatmulGemvShape1MetalKernel),
    Shape2(MatmulGemvShape2MetalKernel),
    Shape3(MatmulGemvShape3MetalKernel),
    Shape4(MatmulGemvShape4MetalKernel),
    Shape5(MatmulGemvShape5MetalKernel),
    Shape6(MatmulGemvShape6MetalKernel),
}

macro_rules! encode_pipeline {
    ($kernel:expr, $matrix:expr, $input_vector:expr, $output_source:expr, $output_vector:expr, $descriptor:expr, $vector_batch_stride:expr, $matrix_batch_stride:expr, $output_source_batch_stride:expr, $encoder:expr) => {{
        $kernel.encode(
            $matrix,
            $input_vector,
            $output_source,
            $output_vector,
            $descriptor.input_dimension,
            $descriptor.output_dimension,
            $descriptor.matrix_leading_dim,
            $descriptor.alpha,
            $descriptor.beta,
            $descriptor.batch_ndim,
            &$descriptor.batch_shape,
            $vector_batch_stride,
            $matrix_batch_stride,
            $output_source_batch_stride,
            $descriptor.bias_stride,
            $descriptor.batch_rows,
            $descriptor.output_ld,
            $descriptor.vector_ld,
            $encoder,
        );
    }};
}

impl DslPipeline {
    fn new(
        context: &MetalContext,
        data_type: DataType,
        key: PipelineKey,
    ) -> Result<Self, MetalError> {
        Ok(match key.shape {
            GemvShape::Shape0 => Self::Shape0(MatmulGemvShape0MetalKernel::new(
                context,
                data_type,
                key.apply_output_scale_and_accumulate,
            )?),
            GemvShape::Shape1 => Self::Shape1(MatmulGemvShape1MetalKernel::new(
                context,
                data_type,
                key.apply_output_scale_and_accumulate,
            )?),
            GemvShape::Shape2 => Self::Shape2(MatmulGemvShape2MetalKernel::new(
                context,
                data_type,
                key.apply_output_scale_and_accumulate,
            )?),
            GemvShape::Shape3 => Self::Shape3(MatmulGemvShape3MetalKernel::new(
                context,
                data_type,
                key.apply_output_scale_and_accumulate,
            )?),
            GemvShape::Shape4 => Self::Shape4(MatmulGemvShape4MetalKernel::new(
                context,
                data_type,
                key.apply_output_scale_and_accumulate,
            )?),
            GemvShape::Shape5 => Self::Shape5(MatmulGemvShape5MetalKernel::new(
                context,
                data_type,
                key.apply_output_scale_and_accumulate,
            )?),
            GemvShape::Shape6 => Self::Shape6(MatmulGemvShape6MetalKernel::new(
                context,
                data_type,
                key.apply_output_scale_and_accumulate,
            )?),
        })
    }

    fn encode(
        &self,
        matrix: &Retained<ProtocolObject<dyn MTLBuffer>>,
        matrix_offset: usize,
        input_vector: &Retained<ProtocolObject<dyn MTLBuffer>>,
        input_vector_offset: usize,
        output_source: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        output_vector: &Retained<ProtocolObject<dyn MTLBuffer>>,
        vector_batch_stride: &[i32; 1],
        matrix_batch_stride: &[i32; 1],
        output_source_batch_stride: &[i32; 1],
        descriptor: &DispatchDescriptor,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    ) {
        let matrix = (matrix, matrix_offset);
        let input_vector = (input_vector, input_vector_offset);
        let output_source = output_source.map(|buffer| (buffer, 0usize));

        match self {
            Self::Shape0(kernel) => {
                encode_pipeline!(
                    kernel,
                    matrix,
                    input_vector,
                    output_source,
                    output_vector,
                    descriptor,
                    vector_batch_stride,
                    matrix_batch_stride,
                    output_source_batch_stride,
                    encoder
                );
            },
            Self::Shape1(kernel) => {
                encode_pipeline!(
                    kernel,
                    matrix,
                    input_vector,
                    output_source,
                    output_vector,
                    descriptor,
                    vector_batch_stride,
                    matrix_batch_stride,
                    output_source_batch_stride,
                    encoder
                );
            },
            Self::Shape2(kernel) => {
                encode_pipeline!(
                    kernel,
                    matrix,
                    input_vector,
                    output_source,
                    output_vector,
                    descriptor,
                    vector_batch_stride,
                    matrix_batch_stride,
                    output_source_batch_stride,
                    encoder
                );
            },
            Self::Shape3(kernel) => {
                encode_pipeline!(
                    kernel,
                    matrix,
                    input_vector,
                    output_source,
                    output_vector,
                    descriptor,
                    vector_batch_stride,
                    matrix_batch_stride,
                    output_source_batch_stride,
                    encoder
                );
            },
            Self::Shape4(kernel) => {
                encode_pipeline!(
                    kernel,
                    matrix,
                    input_vector,
                    output_source,
                    output_vector,
                    descriptor,
                    vector_batch_stride,
                    matrix_batch_stride,
                    output_source_batch_stride,
                    encoder
                );
            },
            Self::Shape5(kernel) => {
                encode_pipeline!(
                    kernel,
                    matrix,
                    input_vector,
                    output_source,
                    output_vector,
                    descriptor,
                    vector_batch_stride,
                    matrix_batch_stride,
                    output_source_batch_stride,
                    encoder
                );
            },
            Self::Shape6(kernel) => {
                encode_pipeline!(
                    kernel,
                    matrix,
                    input_vector,
                    output_source,
                    output_vector,
                    descriptor,
                    vector_batch_stride,
                    matrix_batch_stride,
                    output_source_batch_stride,
                    encoder
                );
            },
        }
    }
}

pub(super) struct DslKernel {
    data_type: DataType,
    pipelines: HashMap<PipelineKey, DslPipeline>,
}

impl DslKernel {
    pub(super) fn new(data_type: DataType) -> Result<Self, MetalError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MetalError::Generic(format!("Unsupported data type for GEMV: {:?}", data_type)));
        }
        Ok(Self {
            data_type,
            pipelines: HashMap::new(),
        })
    }

    pub(super) fn precompile(
        &mut self,
        context: &MetalContext,
    ) -> Result<(), MetalError> {
        let keys: &[PipelineKey] = match self.data_type {
            DataType::BF16 => &[
                PipelineKey {
                    shape: GemvShape::Shape5,
                    apply_output_scale_and_accumulate: false,
                },
                PipelineKey {
                    shape: GemvShape::Shape5,
                    apply_output_scale_and_accumulate: true,
                },
                PipelineKey {
                    shape: GemvShape::Shape6,
                    apply_output_scale_and_accumulate: false,
                },
                PipelineKey {
                    shape: GemvShape::Shape6,
                    apply_output_scale_and_accumulate: true,
                },
            ],
            DataType::F16 => &[PipelineKey {
                shape: GemvShape::Shape6,
                apply_output_scale_and_accumulate: false,
            }],
            DataType::F32 => &[PipelineKey {
                shape: GemvShape::Shape6,
                apply_output_scale_and_accumulate: false,
            }],
            _ => &[],
        };

        for key in keys {
            let _ = self.get_or_create_pipeline(context, *key)?;
        }

        Ok(())
    }

    fn get_or_create_pipeline(
        &mut self,
        context: &MetalContext,
        key: PipelineKey,
    ) -> Result<&DslPipeline, MetalError> {
        if !self.pipelines.contains_key(&key) {
            let pipeline = DslPipeline::new(context, self.data_type, key)?;
            self.pipelines.insert(key, pipeline);
        }
        Ok(self.pipelines.get(&key).unwrap())
    }

    pub(super) fn encode_descriptor(
        &mut self,
        context: &MetalContext,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: &MatmulArguments,
        descriptor: &DispatchDescriptor,
    ) -> Result<bool, MetalError> {
        let shape = GemvShape::from_pipeline_configuration(&descriptor.pipeline_configuration).ok_or_else(|| {
            MetalError::Generic(format!(
                "GEMV DSL path does not support pipeline configuration: {:?}",
                descriptor.pipeline_configuration
            ))
        })?;
        let key = PipelineKey {
            shape,
            apply_output_scale_and_accumulate: descriptor
                .pipeline_configuration
                .apply_output_scale_and_accumulate,
        };
        let pipeline = self.get_or_create_pipeline(context, key)?;

        let (matrix, matrix_offset) = if descriptor.matrix_is_rhs {
            (arguments.b, 0usize)
        } else {
            (arguments.a, arguments.a_offset as usize)
        };
        let (input_vector, input_vector_offset) = if descriptor.matrix_is_rhs {
            (arguments.a, arguments.a_offset as usize)
        } else {
            (arguments.b, 0usize)
        };

        let output_source = if descriptor
            .pipeline_configuration
            .apply_output_scale_and_accumulate
        {
            match descriptor.output_source {
                OutputSource::None => {
                    return Err(MetalError::Generic(
                        "GEMV descriptor mismatch: apply_output_scale_and_accumulate=true but output_source=None"
                            .to_owned(),
                    ));
                },
                OutputSource::Bias => {
                    Some(arguments.bias.ok_or_else(|| {
                        MetalError::Generic("GEMV descriptor requires bias buffer".to_owned())
                    })?)
                },
                OutputSource::C => Some(arguments.c.ok_or_else(|| {
                    MetalError::Generic("GEMV descriptor requires C buffer".to_owned())
                })?),
            }
        } else {
            None
        };

        let vector_batch_stride = [i32::try_from(descriptor.vector_batch_stride[0]).map_err(|_| {
            MetalError::Generic(format!(
                "GEMV DSL path requires i32 vector_batch_stride but got {}",
                descriptor.vector_batch_stride[0]
            ))
        })?];
        let matrix_batch_stride = [i32::try_from(descriptor.matrix_batch_stride[0]).map_err(|_| {
            MetalError::Generic(format!(
                "GEMV DSL path requires i32 matrix_batch_stride but got {}",
                descriptor.matrix_batch_stride[0]
            ))
        })?];
        let output_source_batch_stride = [i32::try_from(descriptor.bias_batch_stride[0]).map_err(|_| {
            MetalError::Generic(format!(
                "GEMV DSL path requires i32 output_source_batch_stride but got {}",
                descriptor.bias_batch_stride[0]
            ))
        })?];

        pipeline.encode(
            matrix,
            matrix_offset,
            input_vector,
            input_vector_offset,
            output_source,
            arguments.d,
            &vector_batch_stride,
            &matrix_batch_stride,
            &output_source_batch_stride,
            descriptor,
            encoder,
        );

        Ok(descriptor.bias_is_fused())
    }
}
