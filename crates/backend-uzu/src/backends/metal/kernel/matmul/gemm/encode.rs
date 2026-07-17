use super::{
    kernel::GemmKernel,
    plan::{GemmEncodeArgs, GemmPlan},
    specialization::GemmSpecialization,
    tiling::split_k_step,
};
use crate::backends::{
    common::{
        Allocation, BufferArg, Encoder,
        gpu_types::{
            GemmParams,
            gemm::{GemmAlignment, GemmBPrologueKind, GemmDTransform},
        },
        kernel::HadamardTransformKernel,
    },
    metal::{Metal, error::MetalError},
};

impl GemmKernel {
    pub(crate) fn encode_with_plan<'a, 'b, 'd, WB: BufferArg<'b, Metal>>(
        &mut self,
        args: GemmEncodeArgs<'a, 'b, 'd, WB>,
        plan: GemmPlan,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        if plan.split_k > 1 {
            self.encode_split_k(args, &plan, encoder)
        } else {
            self.encode_direct(args, &plan, encoder)
        }
    }

    fn encode_direct<'a, 'b, 'd, WB: BufferArg<'b, Metal>>(
        &mut self,
        args: GemmEncodeArgs<'a, 'b, 'd, WB>,
        plan: &GemmPlan,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let specialization = GemmSpecialization {
            weights_data_type: self.weights_data_type,
            tiling: plan.tiling,
            use_mxu: plan.use_mxu,
            output_transform: args.output_transform,
            alignment: plan.alignment,
            transpose_b: plan.transpose_b,
            b_prologue: args.b_prologue,
            bits_per_b: args.bits_per_b,
            group_size: args.group_size,
            a_prologue: args.a_prologue,
        };
        specialization.validate()?;
        let kernel = self.get_or_create(encoder.context(), specialization)?;
        kernel.encode(
            args.a.map(|values| (values, args.a_offset)),
            args.weights,
            &mut *args.d,
            args.scales,
            args.biases,
            args.zero_points,
            args.output_bias,
            args.rht_factors,
            args.a_int8,
            args.a_scales,
            args.a_zero_points,
            args.a_row_sums,
            args.b_col_sums,
            std::slice::from_ref(&plan.params),
            plan.group_count_x,
            plan.group_count_y,
            1,
            encoder,
        );
        Ok(())
    }

    fn encode_split_k<'a, 'b, 'd, WB: BufferArg<'b, Metal>>(
        &mut self,
        args: GemmEncodeArgs<'a, 'b, 'd, WB>,
        plan: &GemmPlan,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let full_precision = matches!(args.b_prologue, GemmBPrologueKind::FullPrecision);
        let split_k = plan.split_k;
        let kp = args.k / split_k;
        let k_step = split_k_step(plan.tiling, plan.use_mxu, args.group_size.unwrap_or(0), full_precision).unwrap_or(1);
        let base_gx = args.n.div_ceil(plan.tiling.block_n());
        let base_gy = args.m.div_ceil(plan.tiling.block_m());
        let alignment = GemmAlignment::new(
            args.m.is_multiple_of(plan.tiling.block_m()),
            args.n.is_multiple_of(plan.tiling.block_n()),
            true,
        );
        let part_spec = GemmSpecialization {
            weights_data_type: self.weights_data_type,
            tiling: plan.tiling,
            use_mxu: plan.use_mxu,
            output_transform: GemmDTransform::empty(),
            alignment,
            transpose_b: true,
            b_prologue: args.b_prologue,
            bits_per_b: args.bits_per_b,
            group_size: args.group_size,
            a_prologue: args.a_prologue,
        };
        part_spec.validate()?;

        let elem = (args.m as usize) * (args.n as usize);
        let slice_bytes = elem * self.output_data_type.size_in_bytes();
        let mut temp = encoder.allocate_scratch(split_k as usize * slice_bytes)?;

        let params = GemmParams {
            M: args.m,
            N: args.n,
            K: args.k,
            leading_dimension_a: args.k,
            leading_dimension_b: args.k,
            leading_dimension_d: args.n,
            threadgroups_per_row: base_gx,
            threadgroups_per_column: base_gy,
            aligned_inner_iterations: kp / k_step,
            use_morton: false,
            ab_scale: 1.0,
        };
        let part_kernel = self.get_or_create(encoder.context(), part_spec)?;
        part_kernel.encode(
            args.a.map(|values| (values, args.a_offset)),
            args.weights,
            &mut temp,
            args.scales,
            args.biases,
            args.zero_points,
            None::<&Allocation<Metal>>,
            None::<&Allocation<Metal>>,
            args.a_int8,
            args.a_scales,
            args.a_zero_points,
            args.a_row_sums,
            args.b_col_sums,
            std::slice::from_ref(&params),
            base_gx,
            base_gy,
            split_k,
            encoder,
        );

        debug_assert_eq!(elem % 4, 0, "split-K reduce requires M*N divisible by 4");
        let group_count = ((elem as u32) / 4).div_ceil(256);
        let reduce_transform = args
            .output_transform
            .intersection(GemmDTransform::SCALE | GemmDTransform::ACCUMULATE | GemmDTransform::BIAS);
        let bias_arg = if reduce_transform.contains(GemmDTransform::BIAS) {
            args.output_bias
        } else {
            None
        };
        let scale_arg = if reduce_transform.contains(GemmDTransform::SCALE) {
            Some(args.ab_scale)
        } else {
            None
        };
        let reduce = self.get_or_create_split_k_reduce(encoder.context(), reduce_transform)?;
        reduce.encode(
            (&temp, 0usize),
            &mut *args.d,
            bias_arg,
            elem as u32,
            split_k,
            group_count,
            args.n,
            scale_arg,
            encoder,
        );

        if args.output_transform.contains(GemmDTransform::RHT)
            && let Some(factors) = args.rht_factors
        {
            self.hadamard.encode(&mut *args.d, factors, args.n, args.m, encoder);
        }
        Ok(())
    }
}
