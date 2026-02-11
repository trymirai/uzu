use metal::{MTLBuffer, MTLComputeCommandEncoder};
use objc2::{__framework_prelude::ProtocolObject, Message};

use crate::{
    DataType,
    backends::{
        common::kernel::QKNormKernel,
        metal::{MTLContext, MTLError, kernel::dsl::QKNormMetalKernel},
    },
};

pub struct QKNormBlock {
    kernel: QKNormMetalKernel,
    full_layer: bool,
}

impl QKNormBlock {
    pub fn new(
        context: &MTLContext,
        input_data_type: DataType,
        scales_data_type: DataType,
        output_data_type: DataType,
        accumulation_data_type: DataType,
        full_layer: bool,
    ) -> Result<Self, MTLError> {
        let kernel = QKNormMetalKernel::new(
            context,
            input_data_type,
            scales_data_type,
            output_data_type,
            accumulation_data_type,
        )?;
        Ok(Self {
            kernel,
            full_layer,
        })
    }

    pub fn encode(
        &self,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: &QKNormArguments,
    ) {
        // Determine which contiguous head range to normalize.
        //
        // QKV layout per token: [Q heads][K heads][V heads]
        // We only ever normalize within [Q] and/or [K].
        let (head_offset, head_count): (u32, u32) = match args.target {
            QKNormTarget::QueryHeads => (0, args.num_q_heads as u32),
            QKNormTarget::KeyHeads => (args.num_q_heads as u32, args.num_kv_heads as u32),
            QKNormTarget::Both => (0, (args.num_q_heads + args.num_kv_heads) as u32),
        };
        if args.batch_size <= 0 || args.head_dim <= 0 || head_count == 0 {
            return;
        }

        // One SIMD-group (one threadgroup) per head.
        self.kernel.encode(
            &args.qkv_input_buffer.retain(),
            &args.scales_buffer.retain(),
            &args.qkv_output_buffer.retain(),
            args.batch_size as u32,
            args.num_q_heads as u32,
            args.num_kv_heads as u32,
            args.head_dim as u32,
            args.epsilon,
            args.scale_offset,
            head_offset,
            head_count,
            self.full_layer,
            compute_encoder,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QKNormTarget {
    QueryHeads,
    KeyHeads,
    Both,
}

#[derive(Debug)]
pub struct QKNormArguments<'a> {
    pub qkv_input_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub scales_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub qkv_output_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub batch_size: i32,
    pub num_q_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub epsilon: f32,
    pub scale_offset: f32,
    pub target: QKNormTarget,
}
