use std::collections::{HashMap, hash_map::Entry};

use super::{
    plan::{GemmPlan, resolve_full_precision, resolve_quant, select_mxu_tiling_for_args, validate_layout},
    specialization::GemmSpecialization,
};
use crate::{
    backends::{
        common::{
            Backend, BufferArg, Encoder,
            gpu_types::{
                HadamardTransformOrder,
                gemm::{GemmDTransform, GemmTiling},
            },
            kernel::{
                HadamardTransformKernel, Kernels, TensorAddBiasKernel,
                matmul::{MatmulArguments, MatmulB},
            },
        },
        metal::{
            Metal,
            context::MetalContext,
            error::MetalError,
            kernel::{GemmMetalKernel, GemmSplitKReduceMetalKernel, TensorAddBiasMetalKernel},
            metal_extensions::DeviceExt,
        },
    },
    data_type::DataType,
};

#[derive(Debug, Clone, Copy)]
pub enum GemmDispatchPath {
    Simdgroup,
    Mxu,
}

pub struct GemmKernel {
    pub(crate) weights_data_type: DataType,
    input_data_type: DataType,
    pub(crate) output_data_type: DataType,
    kernels: HashMap<GemmSpecialization, GemmMetalKernel>,
    pub bias_add: TensorAddBiasMetalKernel,
    pub hadamard: <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel,
    split_k_reduce: HashMap<GemmDTransform, GemmSplitKReduceMetalKernel>,
}

impl GemmKernel {
    pub(crate) fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MetalError> {
        let bias_add = TensorAddBiasMetalKernel::new(context, output_data_type, weights_data_type, true)?;
        let hadamard = <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(
            context,
            output_data_type,
            HadamardTransformOrder::Output,
        )?;
        let kernel = Self {
            weights_data_type,
            input_data_type,
            output_data_type,
            kernels: HashMap::new(),
            bias_add,
            hadamard,
            split_k_reduce: HashMap::new(),
        };
        Ok(kernel)
    }

    pub(crate) fn get_or_create(
        &mut self,
        context: &MetalContext,
        specialization: GemmSpecialization,
    ) -> Result<&GemmMetalKernel, MetalError> {
        match self.kernels.entry(specialization) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemmMetalKernel::new(
                    context,
                    self.input_data_type,
                    self.weights_data_type,
                    self.output_data_type,
                    specialization.tiling,
                    specialization.transpose_b,
                    specialization.use_mxu,
                    specialization.b_prologue,
                    specialization.bits_per_b.unwrap_or(0),
                    specialization.group_size.unwrap_or(0),
                    specialization.a_prologue,
                    specialization.output_transform,
                    specialization.alignment,
                )?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub(crate) fn get_or_create_split_k_reduce(
        &mut self,
        context: &MetalContext,
        output_transform: GemmDTransform,
    ) -> Result<&GemmSplitKReduceMetalKernel, MetalError> {
        match self.split_k_reduce.entry(output_transform) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let kernel = GemmSplitKReduceMetalKernel::new(context, self.output_data_type, output_transform)?;
                Ok(entry.insert(kernel))
            },
        }
    }

    pub(crate) fn should_skip_gemv_for_mxu<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &self,
        arguments: &MatmulArguments<'a, 'b, 'd, Metal, TB>,
    ) -> bool {
        match (
            arguments.m,
            arguments.n == arguments.k,
            (self.weights_data_type, self.input_data_type, self.output_data_type),
        ) {
            (4, true, (DataType::F32, DataType::F32, DataType::F32))
            | (5, _, (DataType::BF16, DataType::BF16, DataType::BF16)) => return false,
            _ => {},
        }
        match arguments.m {
            0..=3 => return false,
            4 => {
                // The M4 MXU tile only uses a quarter of its rows; avoid it for wide-N shapes.
                let small_enough_for_mxu = arguments.n <= 6144 && arguments.k <= 9728;
                let k_dominates = arguments.k > 3_u32.saturating_mul(arguments.n);
                if !(small_enough_for_mxu || k_dominates) {
                    return false;
                }
            },
            _ => {},
        }
        matches!(
            select_mxu_tiling_for_args(arguments, self.weights_data_type, self.input_data_type, self.output_data_type,),
            Some(GemmTiling::Tile16x32x256_Simdgroups1x1 | GemmTiling::Tile16x128x256_Simdgroups1x4)
        )
    }

    pub fn encode<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let path = if encoder.context().device.supports_mxu()
            && (arguments.a.is_int8()
                || select_mxu_tiling_for_args(
                    &arguments,
                    self.weights_data_type,
                    self.input_data_type,
                    self.output_data_type,
                )
                .is_some())
        {
            GemmDispatchPath::Mxu
        } else {
            GemmDispatchPath::Simdgroup
        };
        self.encode_dispatch_path(arguments, path, encoder)
    }

    pub fn encode_dispatch_path<'a, 'b, 'd, TB: BufferArg<'b, Metal>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Metal, TB>,
        path: GemmDispatchPath,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        if matches!(path, GemmDispatchPath::Mxu) {
            assert!(
                encoder.context().device.supports_mxu(),
                "GemmDispatchPath::Mxu requested on hardware without MXU support",
            );
            assert!(
                [self.weights_data_type, self.input_data_type, self.output_data_type]
                    .into_iter()
                    .all(|data_type| matches!(data_type, DataType::BF16 | DataType::F32)),
                "GemmDispatchPath::Mxu requires BF16 or F32 data types, got weights {:?}, input {:?}, output {:?}",
                self.weights_data_type,
                self.input_data_type,
                self.output_data_type,
            );
        }

        validate_layout(&arguments)?;
        let use_mxu = matches!(path, GemmDispatchPath::Mxu);
        let is_quant = !matches!(arguments.b, MatmulB::FullPrecision { .. });

        if is_quant {
            let args = resolve_quant(arguments, use_mxu)?;
            let plan = GemmPlan::build_quant(&args, use_mxu, self.weights_data_type, self.output_data_type);
            self.encode_with_plan(args, plan, encoder)
        } else {
            let args = resolve_full_precision(arguments)?;
            let plan = GemmPlan::build_full_precision(&args, use_mxu, self.weights_data_type, self.output_data_type);
            self.encode_with_plan(args, plan, encoder)
        }
    }
}
