use std::{cell::RefCell, collections::HashMap};

use objc2::Message;

use crate::{
    DataType,
    backends::{
        common::{
            gpu_types::{AttnMaskParams, AttnParams},
            kernel::AttentionGemmKernel,
        },
        metal::{
            MTLBuffer, MTLComputeCommandEncoder, MTLContext, MTLError, ProtocolObject,
            kernel::dsl::AttentionGemmMetalKernel,
        },
    },
};

const BQ: usize = 32;

pub struct AttentionGemmArguments<'a> {
    pub queries_buffer: &'a ProtocolObject<dyn MTLBuffer>, // buffer(0)
    pub keys_buffer: &'a ProtocolObject<dyn MTLBuffer>,    // buffer(1)
    pub values_buffer: &'a ProtocolObject<dyn MTLBuffer>,  // buffer(2)
    pub output_buffer: &'a ProtocolObject<dyn MTLBuffer>,  // buffer(3)
    pub mask_buffer: Option<&'a ProtocolObject<dyn MTLBuffer>>, // buffer(6)
    pub sinks_buffer: Option<&'a ProtocolObject<dyn MTLBuffer>>, // buffer(7)
    pub num_heads: usize,
    pub num_groups: usize,
    pub suffix_length: usize,         // qL
    pub sequence_length: usize,       // kL (prefix + suffix)
    pub segment_prefix_length: usize, // qL_off
    pub max_sequence_length: usize,   // stride for K/V cache
    pub head_dim: usize,
    pub is_causal: bool,
    pub scale: f32,
}

pub struct AttentionGemmBlock {
    data_type: DataType,
    cache: RefCell<HashMap<KernelKey, AttentionGemmMetalKernel>>,
}

impl AttentionGemmBlock {
    pub fn new(data_type: DataType) -> Self {
        let cache = RefCell::new(HashMap::new());
        Self {
            data_type,
            cache,
        }
    }

    pub fn encode(
        &self,
        context: &MTLContext,
        compute_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        args: &AttentionGemmArguments,
    ) -> Result<(), MTLError> {
        let bk: usize = if args.head_dim < 128 {
            32
        } else {
            16
        };
        let align_q = (args.suffix_length % BQ) == 0;
        let align_k = (args.sequence_length % bk) == 0;
        let has_mask = args.mask_buffer.is_some();
        let has_sinks = args.sinks_buffer.is_some();
        let key = KernelKey {
            bk,
            head_dim: args.head_dim,
            align_q,
            align_k,
            is_causal: args.is_causal,
            has_mask,
            has_sinks,
        };

        let mut map = self.cache.borrow_mut();
        let kernel = map.entry(key).or_insert_with(|| {
            AttentionGemmMetalKernel::new(
                context,
                self.data_type,
                bk as u32,
                args.head_dim as u32,
                align_q,
                align_k,
                args.is_causal,
                has_mask,
                has_sinks,
            )
            .expect("Could not initialize AttentionGemmMetalKernel")
        });

        // Params (all strides in elements)
        let q_head_stride = (args.suffix_length * args.head_dim) as i64;
        let q_seq_stride = args.head_dim as i64;

        let kv_head_stride = (args.max_sequence_length * args.head_dim) as i64;
        let kv_seq_stride = args.head_dim as i64;

        let o_head_stride = args.head_dim as i64;
        let o_seq_stride = (args.num_heads * args.head_dim) as i64;

        let nk = (args.sequence_length + bk - 1) / bk;
        let nq_aligned = args.suffix_length / BQ;
        let nk_aligned = args.sequence_length / bk;

        let params = AttnParams {
            q_strides: [0, q_head_stride, q_seq_stride],
            k_strides: [0, kv_head_stride, kv_seq_stride],
            v_strides: [0, kv_head_stride, kv_seq_stride],
            o_strides: [0, o_head_stride, o_seq_stride],
            gqa_factor: (args.num_heads / args.num_groups) as i32,
            scale: args.scale,
            q_len: args.suffix_length as i32,
            k_len: args.sequence_length as i32,
            q_off: args.segment_prefix_length as i32,
            nq_aligned: nq_aligned as i32,
            q_rem: (args.suffix_length - nq_aligned * BQ) as i32,
            nk: nk as i32,
            nk_aligned: nk_aligned as i32,
            k_rem: (args.sequence_length - nk_aligned * bk) as i32,
        };

        let attn_mask_params = AttnMaskParams {
            // We use a shared bias matrix for all heads/batches.
            m_strides: [0, 0, args.sequence_length as i64],
        };
        let attn_mask_params_opt = args.mask_buffer.map(|_| std::slice::from_ref(&attn_mask_params));

        let mask_buffer_retain = args.mask_buffer.map(|b| b.retain());
        let mask_buffer_opt = mask_buffer_retain.as_ref().map(|b| b);
        let sinks_buffer_retain = args.sinks_buffer.map(|b| b.retain());
        let sinks_buffer_opt = sinks_buffer_retain.as_ref().map(|b| b);

        kernel.encode(
            &args.queries_buffer.retain(),
            &args.keys_buffer.retain(),
            &args.values_buffer.retain(),
            &args.output_buffer.retain(),
            std::slice::from_ref(&params),
            attn_mask_params_opt,
            mask_buffer_opt,
            sinks_buffer_opt,
            args.num_heads as u32,
            args.suffix_length as u32,
            &compute_encoder,
        );

        Ok(())
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct KernelKey {
    bk: usize,
    head_dim: usize,
    align_q: bool,
    align_k: bool,
    is_causal: bool,
    has_mask: bool,
    has_sinks: bool,
}
