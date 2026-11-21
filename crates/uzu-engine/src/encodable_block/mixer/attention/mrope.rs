use crate::{
    backends::common::{Backend, CommandBuffer},
    config::rope::AnyRoPEConfig,
    encodable_block::mixer::attention::rope::RoPEPositions,
};

pub struct MRoPEPositions {
    pub positions: Box<[[u32; 3]]>,
}

impl MRoPEPositions {
    pub fn precalculate<B: Backend>(
        &self,
        rope_config: &AnyRoPEConfig,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<RoPEPositions<B>, B::Error> {
        let head_dim = *rope_config.head_dim();
        let frequency_pairs_per_axis = match rope_config {
            AnyRoPEConfig::MultimodalRoPEConfig(multimodal_config) => multimodal_config.frequency_pairs_per_axis,
            // Without M-RoPE every pair rotates by the temporal position
            _ => {
                return RoPEPositions::new(
                    self.positions.len(),
                    head_dim,
                    |token_index, _| self.positions[token_index][0],
                    command_buffer,
                );
            },
        };

        // `Transformer::new` checks that the frequency pairs cover the RoPE head
        RoPEPositions::new(
            self.positions.len(),
            head_dim,
            |token_index, pair_index| {
                self.positions[token_index][interleaved_axis(pair_index, frequency_pairs_per_axis)]
            },
            command_buffer,
        )
    }
}

/// Axis (0 — t, 1 — h, 2 — w) of frequency pair `pair` in the layout `[T H W T H W ... T T]`:
/// every third pair, while the axis has pairs left
fn interleaved_axis(
    pair: u32,
    frequency_pairs_per_axis: [u32; 3],
) -> usize {
    let [_, height_pairs, width_pairs] = frequency_pairs_per_axis;
    match pair % 3 {
        1 if pair < 3 * height_pairs => 1,
        2 if pair < 3 * width_pairs => 2,
        _ => 0,
    }
}
