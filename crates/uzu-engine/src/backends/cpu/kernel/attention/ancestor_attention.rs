use half::bf16;
use uzu_engine_macros::kernel;

use crate::backends::{
    common::gpu_types::weaver::MetadataIdx,
    cpu::kernel::attention::{
        compute::{Row, quantize},
        rope::apply_rope,
    },
};

#[kernel(AncestorAttention)]
#[variants(HEAD_DIM, 128)]
fn ancestor_attention<const HEAD_DIM: u32>(
    prefix_k_codes: *const i8,
    prefix_k_scales: *const bf16,
    prefix_v_codes: *const u8,
    prefix_v_scales: *const bf16,
    node_k_codes: *mut i8,
    node_k_scales: *mut bf16,
    node_v_codes: *mut u8,
    node_v_scales: *mut bf16,
    current_qkv: *const bf16,
    rope_inverse_frequencies: *const f32,
    node_metadata: *const u32,
    ancestor_indices: *const u32,
    ancestor_counts: *const u32,
    node_indices: *const u32,
    output: *mut bf16,
    rows: u32,
    prefix_length: u32,
    ancestor_stride: u32,
    node_capacity: u32,
    max_depth: u32,
    rope_scale: f32,
    scale: f32,
    #[specialize] num_heads: u32,
    #[specialize] backend_opaque_layout: u32,
) {
    let dim = HEAD_DIM as usize;
    let model_dim = num_heads as usize * dim;
    assert_eq!(backend_opaque_layout, 0);
    for row in 0..rows as usize {
        unsafe {
            let depth = *node_metadata.add(MetadataIdx::Depth as usize * rows as usize + row);
            assert!(depth < max_depth, "node metadata depth must be rope-safe");
            for head in 0..num_heads {
                let qkv = current_qkv.add(row * 3 * model_dim + head as usize * dim);
                let rotate = |component| {
                    (0..dim)
                        .map(|c| {
                            apply_rope(
                                qkv.add(component * model_dim),
                                rope_inverse_frequencies,
                                rope_scale,
                                depth + 1,
                                c,
                                dim,
                            )
                            .to_f32()
                        })
                        .collect()
                };
                let (q, qs) = quantize(rotate(0));
                let (k, ks) = quantize(rotate(1));
                let (v, vs) = quantize((0..dim).map(|c| (*qkv.add(2 * model_dim + c)).to_f32()).collect());
                let ancestors = *ancestor_counts.add(row);
                let mut result = Row::new(dim, None);
                for index in 0..prefix_length + ancestors + 1 {
                    let own = index == prefix_length + ancestors;
                    let prefix = index < prefix_length;
                    let slot = if prefix || own {
                        index
                    } else {
                        *ancestor_indices.add(row * ancestor_stride as usize + (index - prefix_length) as usize)
                    };
                    if !prefix && !own {
                        assert!(slot < node_capacity);
                    }
                    let kc = if prefix {
                        prefix_k_codes
                    } else {
                        node_k_codes
                    };
                    let vc = if prefix {
                        prefix_v_codes
                    } else {
                        node_v_codes
                    };
                    let kv_row = (slot * num_heads + head) as usize;
                    let key_scale = if own {
                        ks
                    } else {
                        *(if prefix {
                            prefix_k_scales
                        } else {
                            node_k_scales
                        })
                        .add(kv_row)
                    }
                    .to_f32();
                    let value_scale = if own {
                        vs
                    } else {
                        *(if prefix {
                            prefix_v_scales
                        } else {
                            node_v_scales
                        })
                        .add(kv_row)
                    }
                    .to_f32();
                    let dot: i32 = (0..dim)
                        .map(|c| {
                            i32::from(q[c])
                                * i32::from(if own {
                                    k[c]
                                } else {
                                    *kc.add(kv_row * dim + c)
                                })
                        })
                        .sum();
                    result.add(dot as f32 * qs.to_f32() * key_scale * scale, |c| {
                        (if own {
                            i32::from(v[c])
                        } else {
                            i32::from(*vc.add(kv_row * dim + c)) - 128
                        }) as f32
                            * value_scale
                    });
                }
                if node_capacity > 0 {
                    let slot = *node_indices.add(row);
                    assert!(slot < node_capacity);
                    let kv_row = (slot * num_heads + head) as usize;
                    for c in 0..dim {
                        *node_k_codes.add(kv_row * dim + c) = k[c];
                        *node_v_codes.add(kv_row * dim + c) = (i16::from(v[c]) + 128) as u8;
                    }
                    *node_k_scales.add(kv_row) = ks;
                    *node_v_scales.add(kv_row) = vs;
                }
                for (c, value) in result.finish().into_iter().enumerate() {
                    *output.add((row * num_heads as usize + head as usize) * dim + c) = bf16::from_f32(value);
                }
            }
        }
    }
}
