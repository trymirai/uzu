pub mod gemm;
pub mod gemv;

pub use self::gemm::{GemmDispatchPath, GemmKernel};
use self::gemv::{GemvDispatch, GemvSpecialization};
use crate::{
    backends::{
        common::{
            AsBufferRangeRef, Buffer, Encoder,
            gpu_types::QuantizationMode,
            kernel::matmul::{MatmulArguments, MatmulB, MatmulError, MatmulKernel, MatmulQuantCombo},
        },
        metal::{Metal, context::MetalContext, error::MetalError},
    },
    data_type::DataType,
};

pub struct MatmulMetalKernel {
    gemv: GemvDispatch,
    pub gemm: GemmKernel,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
}

impl MatmulKernel for MatmulMetalKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MetalError> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::BF16 | DataType::F32) {
                return Err(MatmulError::<Metal>::UnsupportedDataType(data_type).into());
            }
        }

        let gemm = GemmKernel::new(context, weights_data_type, input_data_type, output_data_type)?;
        let gemv = GemvDispatch::new(context, weights_data_type, input_data_type, output_data_type)
            .map_err(MetalError::from)?;

        Ok(Self {
            gemv,
            gemm,
            weights_data_type,
            input_data_type,
            output_data_type,
        })
    }

    fn encode<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        validate_lloyd_max_qmv(&arguments)?;
        match GemvSpecialization::select(
            &arguments,
            self.weights_data_type,
            self.input_data_type,
            self.output_data_type,
        ) {
            Some(spec) => self.gemv.encode(arguments, spec, encoder).map_err(MetalError::from),
            None => self.gemm.encode(arguments, encoder),
        }
    }

    fn preheat_quant_combo(
        &mut self,
        context: &MetalContext,
        combo: MatmulQuantCombo,
    ) -> Result<(), MetalError> {
        self.gemv.preheat_quant_combo(context, combo).map_err(MetalError::from)?;
        self.gemm.preheat_quant_combo(context, combo)
    }
}

fn validate_lloyd_max_qmv<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
    arguments: &MatmulArguments<Metal, TB>
) -> Result<(), MetalError> {
    let MatmulB::LloydMaxDequant {
        mode,
        group_size,
        ..
    } = &arguments.b
    else {
        return Ok(());
    };

    if *mode != QuantizationMode::U4 {
        return Err(unsupported_lloyd_max("only U4 quantization mode is supported"));
    }

    if !matches!(*group_size, 16 | 32 | 64 | 128) {
        return Err(MatmulError::<Metal>::UnsupportedGroupSize(*group_size as usize).into());
    }

    if !arguments.b_transpose || arguments.b_offset != 0 || arguments.b_leading_dimension.is_some() {
        return Err(MatmulError::<Metal>::UnsupportedLayout {
            path: "Lloyd-Max QMV",
        }
        .into());
    }

    if arguments.d_transform.rht_factors.is_some() {
        return Err(unsupported_lloyd_max("RHT output transform is not implemented"));
    }

    if arguments.m >= 5 {
        return Err(unsupported_lloyd_max("only decode batches with m < 5 are supported"));
    }

    if arguments.n < 4 || !arguments.n.is_multiple_of(32) {
        return Err(unsupported_lloyd_max("output width must be at least 4 and a multiple of 32"));
    }

    if !arguments.k.is_multiple_of(512) {
        return Err(unsupported_lloyd_max("input width must be a multiple of 512"));
    }

    Ok(())
}

fn unsupported_lloyd_max(reason: &'static str) -> MetalError {
    MatmulError::<Metal>::UnsupportedFeature {
        feature: "Lloyd-Max QMV",
        reason,
    }
    .into()
}
