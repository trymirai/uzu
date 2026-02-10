use crate::backends::metal::{KernelDataType, MTLBuffer, ProtocolObject};

#[derive(Debug)]
pub struct MoeExpertsTwoPassArguments<'a> {
    pub x_perm_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [sum_k, d_model] - permuted input
    pub expert_offsets: &'a ProtocolObject<dyn MTLBuffer>, // [E+1] - expert segment offsets
    pub row_expert_map: &'a ProtocolObject<dyn MTLBuffer>,
    pub hidden_buffer: &'a ProtocolObject<dyn MTLBuffer>,
    pub output_buffer: &'a ProtocolObject<dyn MTLBuffer>, // [sum_k, d_model] - output buffer
    pub w13_all: &'a ProtocolObject<dyn MTLBuffer>, // [E, 2*d_ff, d_model] - transposed up projection weights
    pub w2_all: &'a ProtocolObject<dyn MTLBuffer>, // [E, d_model, d_ff] - transposed down projection weights
    pub up_biases: &'a ProtocolObject<dyn MTLBuffer>, // [E, 2*d_ff] - up projection biases
    pub down_biases: &'a ProtocolObject<dyn MTLBuffer>, // [E, d_model] - down projection biases
    pub tile_counts: &'a ProtocolObject<dyn MTLBuffer>, // [E]
    pub tile_offsets: &'a ProtocolObject<dyn MTLBuffer>, // [E+1]
    pub tile_map: &'a ProtocolObject<dyn MTLBuffer>,    // [max_tiles * 3]
    pub total_tiles: &'a ProtocolObject<dyn MTLBuffer>, // [2]
    pub dispatch_args: &'a ProtocolObject<dyn MTLBuffer>, // [3]
    pub total_rows: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub e: usize,
    pub num_tiles_k: u32,
    pub gating_code: u32,
    pub gate_clip_min: f32,
    pub gate_clip_max: f32,
    pub up_clip_min: f32,
    pub up_clip_max: f32,
    pub silu_alpha: f32,
    pub data_type: KernelDataType,
}
