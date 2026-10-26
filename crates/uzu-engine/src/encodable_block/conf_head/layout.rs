/// Offsets, in f32 elements, of the local part's weights inside the head's packed buffer.
///
/// The local part of a slot reads its pool set encoder, the standardization of the slot numbers, the edge MLP and
/// the alpha/beta outputs; all of them are small, so the head packs them into one buffer at load and the kernels
/// index it with these offsets. The Metal kernels compute the same layout in `confidence_head_layout.h`; the two must stay
/// in this order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalLayout {
    pub pool_entry_weights: usize,
    pub pool_entry_biases: usize,
    pub pool_hidden_weights: usize,
    pub pool_hidden_biases: usize,
    pub pool_out_weights: usize,
    pub pool_out_biases: usize,
    pub slot_means: usize,
    pub slot_stds: usize,
    pub slot_keep: usize,
    pub edge_input_scales: usize,
    pub edge_input_biases: usize,
    pub edge_entry_weights: usize,
    pub edge_entry_biases: usize,
    pub edge_block_scales: usize,
    pub edge_block_biases: usize,
    pub edge_up_weights: usize,
    pub edge_up_biases: usize,
    pub edge_down_weights: usize,
    pub edge_down_biases: usize,
    pub edge_final_scales: usize,
    pub edge_final_biases: usize,
    pub delta_weights: usize,
    pub delta_bias: usize,
    pub output_weights: usize,
    pub output_biases: usize,
    pub total: usize,
}

use super::{
    CONF_HEAD_NODE_OUTPUTS as NODE_OUTPUTS, CONF_HEAD_POOL_INPUTS as POOL_INPUTS,
    CONF_HEAD_SLOT_NUMBERS as SLOT_NUMBERS,
};

impl LocalLayout {
    /// The edge MLP's input: [z_u; e_c; z_u * e_c; slot].
    pub fn edge_input(hidden: usize) -> usize {
        3 * hidden + SLOT_NUMBERS
    }

    pub fn new(
        hidden: usize,
        pool_hidden: usize,
        edge_hidden: usize,
    ) -> Self {
        let edge_input = Self::edge_input(hidden);
        let sizes = [
            pool_hidden * POOL_INPUTS,
            pool_hidden,
            pool_hidden * pool_hidden,
            pool_hidden,
            hidden * 2 * pool_hidden,
            hidden,
            SLOT_NUMBERS,
            SLOT_NUMBERS,
            SLOT_NUMBERS,
            edge_input,
            edge_input,
            hidden * edge_input,
            hidden,
            hidden,
            hidden,
            edge_hidden * hidden,
            edge_hidden,
            hidden * edge_hidden,
            hidden,
            hidden,
            hidden,
            hidden,
            1,
            NODE_OUTPUTS * hidden,
            NODE_OUTPUTS,
        ];
        let mut offsets = [0usize; 26];
        for (index, size) in sizes.iter().enumerate() {
            offsets[index + 1] = offsets[index] + size;
        }
        Self {
            pool_entry_weights: offsets[0],
            pool_entry_biases: offsets[1],
            pool_hidden_weights: offsets[2],
            pool_hidden_biases: offsets[3],
            pool_out_weights: offsets[4],
            pool_out_biases: offsets[5],
            slot_means: offsets[6],
            slot_stds: offsets[7],
            slot_keep: offsets[8],
            edge_input_scales: offsets[9],
            edge_input_biases: offsets[10],
            edge_entry_weights: offsets[11],
            edge_entry_biases: offsets[12],
            edge_block_scales: offsets[13],
            edge_block_biases: offsets[14],
            edge_up_weights: offsets[15],
            edge_up_biases: offsets[16],
            edge_down_weights: offsets[17],
            edge_down_biases: offsets[18],
            edge_final_scales: offsets[19],
            edge_final_biases: offsets[20],
            delta_weights: offsets[21],
            delta_bias: offsets[22],
            output_weights: offsets[23],
            output_biases: offsets[24],
            total: offsets[25],
        }
    }
}
