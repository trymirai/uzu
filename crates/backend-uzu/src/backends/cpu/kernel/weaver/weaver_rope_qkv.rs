use half::bf16;
use num_traits::Float;
use proc_macros::kernel;

use crate::{array::ArrayElement, backends::common::gpu_types::weaver::MetadataIdx};

#[kernel(WeaverRopeQkv)]
#[variants(ElementT, bf16)]
#[variants(RopeT, f32)]
pub fn weaver_rope_qkv<ElementT: ArrayElement + Float, RopeT: ArrayElement + Float>(
    qkv: *mut ElementT,
    cosines: *const RopeT,
    sines: *const RopeT,
    node_metadata: *const u32,
    num_heads: u32,
    head_dim: u32,
    max_depth: u32,
    rows: u32,
) {
    const QUERY_AND_KEY_COMPONENTS: usize = 2;
    assert!(head_dim > 0 && head_dim.is_multiple_of(2), "weaver rope head_dim must be positive and even");
    assert!(max_depth > 0, "weaver rope requires a positive max_depth");

    let num_heads = num_heads as usize;
    let head_dim = head_dim as usize;
    let rows = rows as usize;
    let half_dim = head_dim / 2;
    let model_dim = num_heads * head_dim;
    let qkv_width = 3 * model_dim;

    for row in 0..rows {
        let depth = unsafe { *node_metadata.add(MetadataIdx::Depth as usize * rows + row) };
        let position = depth.min(max_depth - 1) as usize + 1;
        for head in 0..QUERY_AND_KEY_COMPONENTS * num_heads {
            let head_base = row * qkv_width + head * head_dim;
            for pair in 0..half_dim {
                unsafe {
                    let low = qkv.add(head_base + pair);
                    let high = qkv.add(head_base + half_dim + pair);
                    let low_value = (*low).to_f32().unwrap();
                    let high_value = (*high).to_f32().unwrap();
                    let rope_index = position * head_dim + pair;
                    let low_cosine = (*cosines.add(rope_index)).to_f32().unwrap();
                    let low_sine = (*sines.add(rope_index)).to_f32().unwrap();
                    let high_cosine = (*cosines.add(rope_index + half_dim)).to_f32().unwrap();
                    let high_sine = (*sines.add(rope_index + half_dim)).to_f32().unwrap();
                    *low = ElementT::from(low_value * low_cosine - high_value * low_sine).unwrap();
                    *high = ElementT::from(high_value * high_cosine + low_value * high_sine).unwrap();
                }
            }
        }
    }
}
