use crate::{
    backends::common::{Backend, CommandBuffer, CommandBufferEncoding},
    config::rope::AnyRoPEConfig,
    encodable_block::mixer::attention::rope::PrecalculatedRoPE,
};

pub struct MRoPEPositions {
    pub positions: Box<[[u32; 3]]>,
}

impl MRoPEPositions {
    pub fn precalculate<B: Backend>(
        &self,
        rope_config: &AnyRoPEConfig,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<PrecalculatedRoPE<B>, B::Error> {
        let frequency_pairs_per_axis = match rope_config {
            AnyRoPEConfig::MultimodalRoPEConfig(multimodal_config) => multimodal_config.frequency_pairs_per_axis,
            _ => {
                let temporal_positions = self.positions.iter().map(|[t, _, _]| *t).collect::<Box<[u32]>>();
                return PrecalculatedRoPE::precalculate(rope_config, &temporal_positions, command_buffer);
            },
        };

        let head_dim = *rope_config.head_dim() as usize;
        let half_dim = head_dim / 2;
        assert_eq!(
            frequency_pairs_per_axis.iter().sum::<u32>() as usize,
            half_dim,
            "M-RoPE frequency pairs don't cover the RoPE head"
        );

        let mut cosines = vec![0.0; self.positions.len() * head_dim];
        let mut sines = vec![0.0; self.positions.len() * head_dim];
        for pair in 0..half_dim {
            let inverse_frequency = 1.0 / rope_config.base().powf((2 * pair) as f32 / head_dim as f32);
            let axis = interleaved_axis(pair as u32, frequency_pairs_per_axis);
            for (token, position) in self.positions.iter().enumerate() {
                let angle = position[axis] as f32 * inverse_frequency;
                let sine = angle.sin();
                let cosine = angle.cos();
                for channel in [token * head_dim + pair, token * head_dim + pair + half_dim] {
                    cosines[channel] = cosine;
                    sines[channel] = sine;
                }
            }
        }

        Ok(PrecalculatedRoPE {
            cosines: command_buffer.allocate_constant_from_slice(&cosines)?,
            sines: command_buffer.allocate_constant_from_slice(&sines)?,
            dim: head_dim as u32,
        })
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
