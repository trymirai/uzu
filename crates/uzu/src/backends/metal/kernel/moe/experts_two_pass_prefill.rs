use std::ptr::NonNull;

use metal::{
    MTLCommandBuffer, MTLCommandEncoder,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDataType,
    MTLFunctionConstantValues, MTLSize,
};
use objc2::{
    __framework_prelude::{ProtocolObject, Retained},
    Message,
};

use crate::backends::metal::kernel::moe::dtype_suffix;
use crate::backends::{
    common::kernel::{
        // MoeTwoPassPrefillPassAIndirectKernel,
        MoeTwoPassPrefillPassBIndirectKernel,
    },
    metal::{
        KernelDataType, MTLContext, MTLError,
        kernel::{
            MoeExpertsTwoPassArguments,
            dsl::{
                // MoeTwoPassPrefillPassAIndirectMetalKernel,
                MoeTwoPassPrefillPassBIndirectMetalKernel,
            },
            moe::tiles_map::{
                MoeTileCountsArguments, MoeTileDispatchArguments,
                MoeTileMapBuildArguments, MoeTileMapKernels,
                MoeTileScanArguments,
            },
        },
    },
};

static DTYPES: [KernelDataType; 3] = [
    KernelDataType::Float16,
    KernelDataType::BFloat16,
    KernelDataType::Float32,
];

pub struct MoeExpertsTwoPassPrefillKernels {
    tile_map: MoeTileMapKernels,
    // passes_a: Vec<Vec<MoeTwoPassPrefillPassAIndirectMetalKernel>>,  // [gate][dtype]
    pass_a_indirect:
        Vec<Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>>, // [gate][dtype]
    passes_b: Vec<MoeTwoPassPrefillPassBIndirectMetalKernel>, // [dtype]
}

impl MoeExpertsTwoPassPrefillKernels {
    pub fn new(ctx: &MTLContext) -> Result<Self, MTLError> {
        // let mut passes_a = Vec::with_capacity(4);
        // for gate in 0..4 {
        //     let mut gate_kernels = Vec::with_capacity(DTYPES.len());
        //     for dtype in DTYPES {
        //         let kernel = MoeTwoPassPrefillPassAIndirectMetalKernel::new(
        //             ctx,
        //             dtype.into(),
        //             gate,
        //         )?;
        //         gate_kernels.push(kernel)
        //     }
        //     passes_a.push(gate_kernels);
        // }

        let mut passes_b = Vec::with_capacity(DTYPES.len());
        for dtype in DTYPES {
            let kernel = MoeTwoPassPrefillPassBIndirectMetalKernel::new(
                ctx,
                dtype.into(),
            )?;
            passes_b.push(kernel);
        }

        let mut pass_a_indirect = vec![Vec::with_capacity(DTYPES.len()); 4];
        for gate in 0u32..4u32 {
            for dtype in &DTYPES {
                let dtype_suffix = dtype_suffix(*dtype);
                let fcv = MTLFunctionConstantValues::new();
                fcv.set_constant_value_type_at_index(
                    NonNull::from(&gate).cast(),
                    MTLDataType::UInt,
                    30,
                );
                let kernel_name = format!(
                    "moe_two_pass_prefill_pass_a_indirect_{}",
                    dtype_suffix
                );
                let cache_key = format!("{}_gate_{}", kernel_name, gate);
                pass_a_indirect[gate as usize].push(
                    ctx.compute_pipeline_state_cached(
                        &cache_key,
                        &kernel_name,
                        Some(&fcv),
                    )?,
                );
            }
        }

        Ok(Self {
            tile_map: MoeTileMapKernels::new(ctx)?,
            // passes_a,
            passes_b,
            pass_a_indirect,
        })
    }

    pub fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        args: &MoeExpertsTwoPassArguments,
    ) {
        if args.total_rows == 0 {
            return;
        }

        let gate_idx = args.gating_code.min(3) as usize;
        let dtype_idx =
            DTYPES.iter().position(|&t| t == args.data_type.into()).unwrap();

        self.tile_map.encode_counts(
            command_buffer,
            &MoeTileCountsArguments {
                offsets_buffer: args.expert_offsets,
                tile_counts_buffer: args.tile_counts,
                e: args.e,
            },
        );
        self.tile_map.encode_scan(
            command_buffer,
            &MoeTileScanArguments {
                tile_counts_buffer: args.tile_counts,
                tile_offsets_buffer: args.tile_offsets,
                total_tiles_buffer: args.total_tiles,
                e: args.e,
            },
        );
        self.tile_map.encode_build_map(
            command_buffer,
            &MoeTileMapBuildArguments {
                expert_offsets: args.expert_offsets,
                tile_offsets: args.tile_offsets,
                tile_counts: args.tile_counts,
                tile_map: args.tile_map,
                e: args.e,
            },
        );
        let d_model_u32 = args.d_model as u32;
        let d_ff_u32 = args.d_ff as u32;
        let e_u32 = args.e as u32;
        const COL_TILE_FF: u32 = 32; // Must match PASSA_BN in Metal kernel
        const COL_TILE_MODEL: u32 = 64;
        let n_tiles_ff = if d_ff_u32 == 0 {
            0
        } else {
            (d_ff_u32 + COL_TILE_FF - 1) / COL_TILE_FF
        };
        let n_tiles_model = if d_model_u32 == 0 {
            0
        } else {
            (d_model_u32 + COL_TILE_MODEL - 1) / COL_TILE_MODEL
        };
        self.tile_map.encode_dispatch_args(
            command_buffer,
            &MoeTileDispatchArguments {
                total_tiles: args.total_tiles,
                dispatch_args: args.dispatch_args,
                num_tiles_x: n_tiles_ff,
            },
        );

        // let pass_a_kernel = &self.passes_a[gate_idx][dtype_idx];
        // let encoder_a = command_buffer
        //     .new_compute_command_encoder()
        //     .expect("Failed to create compute command encoder");
        // pass_a_kernel.encode(
        //     &args.x_perm_buffer.retain(),
        //     &args.expert_offsets.retain(),
        //     &args.w13_all.retain(),
        //     &args.up_biases.retain(),
        //     &args.hidden_buffer.retain(),
        //     args.d_model as u32,
        //     args.d_ff as u32,
        //     args.e as u32,
        //     args.gate_clip_min,
        //     args.gate_clip_max,
        //     args.up_clip_min,
        //     args.up_clip_max,
        //     args.silu_alpha,
        //     &args.tile_map.retain(),
        //     &args.dispatch_args.retain(),
        //     &encoder_a,
        // );
        // encoder_a.end_encoding();

        let pass_a_pipeline = &self.pass_a_indirect[gate_idx][dtype_idx];
        let encoder_a = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        encoder_a.set_compute_pipeline_state(pass_a_pipeline);
        encoder_a.set_buffer(Some(args.x_perm_buffer), 0, 0);
        encoder_a.set_buffer(Some(args.expert_offsets), 0, 1);
        encoder_a.set_buffer(Some(args.w13_all), 0, 2);
        encoder_a.set_buffer(Some(args.up_biases), 0, 3);
        encoder_a.set_buffer(Some(args.hidden_buffer), 0, 4);
        unsafe {
            encoder_a.set_bytes(
                NonNull::new_unchecked(&d_model_u32 as *const _ as *mut _),
                size_of::<u32>(),
                5,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&d_ff_u32 as *const _ as *mut _),
                size_of::<u32>(),
                6,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&e_u32 as *const _ as *mut _),
                size_of::<u32>(),
                7,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(
                    &args.gate_clip_min as *const _ as *mut _,
                ),
                size_of::<f32>(),
                8,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(
                    &args.gate_clip_max as *const _ as *mut _,
                ),
                size_of::<f32>(),
                9,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&args.up_clip_min as *const _ as *mut _),
                size_of::<f32>(),
                10,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&args.up_clip_max as *const _ as *mut _),
                size_of::<f32>(),
                11,
            );
            encoder_a.set_bytes(
                NonNull::new_unchecked(&args.silu_alpha as *const _ as *mut _),
                size_of::<f32>(),
                12,
            );
        }
        encoder_a.set_buffer(Some(args.tile_map), 0, 13);
        // Match kernel config: WM=2, WN=2 => 4 SIMDgroups => 128 threads
        const SIMDGROUPS_PER_TG: u32 = 4;
        const THREADS_PER_TG: u32 = SIMDGROUPS_PER_TG * 32;
        encoder_a.dispatch_threadgroups_indirect(
            args.dispatch_args,
            0,
            MTLSize::new(THREADS_PER_TG as usize, 1, 1),
        );
        encoder_a.end_encoding();

        self.tile_map.encode_dispatch_args(
            command_buffer,
            &MoeTileDispatchArguments {
                total_tiles: args.total_tiles,
                dispatch_args: args.dispatch_args,
                num_tiles_x: n_tiles_model,
            },
        );

        let pass_b_kernel = &self.passes_b[dtype_idx];
        let encoder_b = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        pass_b_kernel.encode(
            &args.hidden_buffer.retain(),
            &args.expert_offsets.retain(),
            &args.w2_all.retain(),
            &args.down_biases.retain(),
            &args.output_buffer.retain(),
            args.d_model as u32,
            args.d_ff as u32,
            args.e as u32,
            &args.tile_map.retain(),
            &args.dispatch_args.retain(),
            &encoder_b,
        );
        encoder_b.end_encoding();
    }
}
