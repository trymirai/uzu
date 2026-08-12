use crate::{
    backends::common::{
        Allocation, Backend, BufferArg, Encoder, Kernels,
        kernel::{AttentionTwoPass1Kernel, AttentionTwoPass2Kernel},
    },
    data_type::DataType,
    encodable_block::mixer::attention::core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments},
};

const INNER_DATA_TYPE: DataType = DataType::F32;

pub struct AttentionTwoPassCore<B: Backend> {
    head_dim: u32,
    num_groups: u32,
    num_q_heads: u32,
    sliding_window_size: Option<u32>,
    scale: Option<f32>,
    data_type: DataType,
    pass_1: <B::Kernels as Kernels>::AttentionTwoPass1Kernel,
    pass_2: <B::Kernels as Kernels>::AttentionTwoPass2Kernel,
}

impl<B: Backend> AttentionTwoPassCore<B> {
    pub fn new(
        arguments: &AttentionCoreNewArguments,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let pass_1 = <B::Kernels as Kernels>::AttentionTwoPass1Kernel::new(
            context,
            arguments.data_type,
            arguments.head_dim,
            arguments.has_sinks,
            arguments.is_kv_cache_ring,
            arguments.is_causal,
            arguments.is_trie,
            arguments.sliding_window_size.is_some(),
        )?;

        let pass_2 =
            <B::Kernels as Kernels>::AttentionTwoPass2Kernel::new(context, arguments.data_type, arguments.head_dim)?;

        Ok(Self {
            head_dim: arguments.head_dim,
            num_groups: arguments.num_groups,
            num_q_heads: arguments.num_q_heads,
            sliding_window_size: arguments.sliding_window_size,
            scale: arguments.scale,
            data_type: arguments.data_type,
            pass_1,
            pass_2,
        })
    }

    pub fn encode<'a, KT: BufferArg<'a, B>, VT: BufferArg<'a, B>>(
        &self,
        arguments: AttentionCoreEncodeArguments<'a, B, KT, VT>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut partials = encoder.allocate_scratch_with_shape(
            &[self.num_q_heads * arguments.suffix_length * 32 * self.head_dim],
            INNER_DATA_TYPE,
        )?;
        let mut sums =
            encoder.allocate_scratch_with_shape(&[self.num_q_heads * arguments.suffix_length * 32], INNER_DATA_TYPE)?;
        let mut maxs =
            encoder.allocate_scratch_with_shape(&[self.num_q_heads * arguments.suffix_length * 32], INNER_DATA_TYPE)?;

        self.pass_1.encode(
            arguments.queries,
            arguments.keys,
            arguments.values,
            &mut partials,
            &mut sums,
            &mut maxs,
            self.num_q_heads / self.num_groups,
            arguments.state_type.physical_prefix_length() + arguments.suffix_length,
            self.head_dim,
            self.num_groups * self.head_dim,
            self.head_dim,
            self.num_groups * self.head_dim,
            arguments.state_type.ring_params(),
            self.scale.unwrap_or(1.0f32 / (self.head_dim as f32).sqrt()),
            self.num_q_heads,
            arguments.suffix_length,
            arguments.trie,
            self.sliding_window_size,
            arguments.sinks,
            encoder,
        );

        let mut output = encoder.allocate_constant_with_shape(
            &[arguments.suffix_length, self.num_q_heads, self.head_dim],
            self.data_type,
        )?;
        self.pass_2.encode(&partials, &sums, &maxs, &mut output, self.num_q_heads, arguments.suffix_length, encoder);

        Ok(output)
    }
}
