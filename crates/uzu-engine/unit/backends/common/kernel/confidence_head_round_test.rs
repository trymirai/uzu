use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::weaver::{FrontierIdx, MetadataIdx, TreeIdx},
            kernel::ConfidenceHeadRoundKernel,
        },
        cpu::Cpu,
    },
    encodable_block::conf_head::layout::LocalLayout,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, for_each_non_cpu_backend},
};

fn log_softmax(values: &[f64]) -> Vec<f64> {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let normalizer = max + values.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    values.iter().map(|x| x - normalizer).collect()
}

fn close(
    actual: f32,
    expected: f64,
) {
    assert!((actual as f64 - expected).abs() < 2e-4, "{actual} != {expected}");
}

fn check_round<B: Backend>() {
    const ROWS: usize = 3;
    const SLOTS: usize = 4;
    const K: usize = 3;
    const C: usize = 2;
    const H: usize = 4;
    const P: usize = 2;
    const FRONTIER: usize = SLOTS * C;
    let layout = LocalLayout::new(H, P, H);
    let mut weights = vec![0.0f32; layout.total];
    // Pool features are SiLU(log p_draft + 0.1) and SiLU(log p_weaver - log p_draft - 0.2).
    weights[layout.pool_entry_weights] = 1.0;
    weights[layout.pool_entry_weights + 4] = 1.0;
    weights[layout.pool_entry_biases..layout.pool_entry_biases + P].copy_from_slice(&[0.1, -0.2]);
    weights[layout.pool_hidden_weights] = 1.0;
    weights[layout.pool_hidden_weights + 3] = 1.0;
    for unit in 0..H {
        weights[layout.pool_out_weights + unit * 2 * P + unit] = 1.0;
    }
    weights[layout.slot_stds..layout.slot_stds + 4].fill(1.0);
    weights[layout.slot_keep..layout.slot_keep + 4].fill(1.0);
    weights[layout.edge_input_scales..layout.edge_input_scales + 3 * H + 4].fill(1.0);
    weights[layout.edge_block_scales..layout.edge_block_scales + H].fill(1.0);
    weights[layout.edge_final_scales..layout.edge_final_scales + H].fill(1.0);
    // A constant delta isolates calibration/normalization from the edge MLP.
    weights[layout.delta_bias] = 0.7;
    weights[layout.output_weights] = 0.2;
    weights[layout.output_weights + H + 1] = -0.1;
    weights[layout.output_biases..layout.output_biases + 2].copy_from_slice(&[0.3, -0.3]);

    let draft = [1.0f32, -0.5, 0.25, 0.0, 0.5, -1.0, -0.25, 1.0, 0.25];
    let residual: Vec<_> =
        [0.25, -0.5, 0.125, 0.0, 0.0, 0.0, -0.25, 0.5, -0.125].into_iter().map(bf16::from_f32).collect();
    let ids = [5u32, 2, 7, 1, 2, 3, 7, 3, 2];
    let noise = [0.1f32, -0.2, 0.3, 0.4, -0.5, 0.6];
    let draft_rows = [0.2f32, -0.3, 0.4, 0.5, -0.1, 0.6, 0.7, -0.8];
    let table: Vec<f32> = (0..8 * H).map(|i| 0.123456 + i as f32 * 0.00123).collect();
    let mut metadata = vec![0u32; MetadataIdx::COUNT * ROWS];
    metadata[MetadataIdx::Depth as usize * ROWS..MetadataIdx::Depth as usize * ROWS + ROWS].copy_from_slice(&[0, 1, 3]); // depth 3 reuses the last pool row
    metadata[MetadataIdx::TreeSlot as usize * ROWS..MetadataIdx::TreeSlot as usize * ROWS + ROWS]
        .copy_from_slice(&[2, 0, 1]); // batch rows and tree slots deliberately differ
    let mut tree = vec![0u32; TreeIdx::COUNT * SLOTS];
    tree[TreeIdx::PathLogprobBits as usize * SLOTS + 2] = (-0.4f32).to_bits();
    tree[TreeIdx::PathLogprobBits as usize * SLOTS + 1] = (-1.2f32).to_bits();
    let mut frontier_values = vec![0u32; FrontierIdx::COUNT * FRONTIER];
    for (slot, children) in [(2, [7, 5]), (1, [3, 2])] {
        for (child, token) in children.into_iter().enumerate() {
            frontier_values[FrontierIdx::TokenId as usize * FRONTIER + slot * C + child] = token;
            frontier_values[FrontierIdx::Active as usize * FRONTIER + slot * C + child] = 1;
        }
    }
    let context = create_context::<B>();
    let residual_buffer = create_buffer_with_data::<B, _>(&context, &residual);
    let draft_buffer = create_buffer_with_data::<B, _>(&context, &draft);
    let ids_buffer = create_buffer_with_data::<B, _>(&context, &ids);
    let noise_buffer = create_buffer_with_data::<B, _>(&context, &noise);
    let draft_rows_buffer = create_buffer_with_data::<B, _>(&context, &draft_rows);
    let metadata_buffer = create_buffer_with_data::<B, _>(&context, &metadata);
    let valid = create_buffer_with_data::<B, _>(&context, &[1u32, 0, 1]);
    let tree_buffer = create_buffer_with_data::<B, _>(&context, &tree);
    let table_buffer = create_buffer_with_data::<B, _>(&context, &table);
    let weights_buffer = create_buffer_with_data::<B, _>(&context, &weights);
    let mut edges = create_buffer_with_data::<B, _>(&context, &[-99f32; ROWS * C]);
    let mut frontier = create_buffer_with_data::<B, _>(&context, &frontier_values);
    let mut nodes = create_buffer_with_data::<B, _>(&context, &[-99f32; SLOTS * H]);
    let mut inputs = create_buffer_with_data::<B, _>(&context, &[-99f32; SLOTS * C * (H + 4)]);
    let mut indices = create_buffer_with_data::<B, _>(&context, &[u32::MAX; SLOTS * C]);
    let mut real = create_buffer_with_data::<B, _>(&context, &[0u32; SLOTS * C]);
    let mut slot_logits = create_buffer_with_data::<B, _>(&context, &[-99f32; SLOTS * K]);
    let kernel = <B::Kernels as Kernels>::ConfidenceHeadRoundKernel::new(
        &context, K as u32, C as u32, H as u32, P as u32, H as u32,
    )
    .unwrap();
    let mut commands = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &residual_buffer,
        &draft_buffer,
        &ids_buffer,
        &noise_buffer,
        &draft_rows_buffer,
        &metadata_buffer,
        &valid,
        &tree_buffer,
        &table_buffer,
        &weights_buffer,
        &mut edges,
        &mut frontier,
        &mut nodes,
        &mut inputs,
        &mut indices,
        &mut real,
        &mut slot_logits,
        ROWS as u32,
        2,
        0.8,
        1e-5,
        FRONTIER as u32,
        SLOTS as u32,
        &mut commands,
    );
    block_on(commands.end_encoding().submit().wait_until_completed()).unwrap();
    let edges = buffer_to_vec::<B, f32>(&edges);
    let frontier = buffer_to_vec::<B, u32>(&frontier);
    let nodes = buffer_to_vec::<B, f32>(&nodes);
    let inputs = buffer_to_vec::<B, f32>(&inputs);
    let indices = buffer_to_vec::<B, u32>(&indices);
    let real = buffer_to_vec::<B, u32>(&real);
    let slot_logits = buffer_to_vec::<B, f32>(&slot_logits);

    for (row, slot, depth, selected, parent) in [(0, 2, 0, [2, 0], -0.4f32), (2, 1, 1, [1, 2], -1.2)] {
        let a: Vec<_> = draft[row * K..(row + 1) * K].iter().map(|x| *x as f64).collect();
        let l: Vec<_> = (0..K).map(|j| a[j] + residual[row * K + j].to_f32() as f64).collect();
        let log_a = log_softmax(&a);
        let log_l = log_softmax(&l);
        let mut summary = [0.0, 0.0, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for j in 0..K {
            for (unit, x) in [log_a[j] + 0.1f32 as f64, log_l[j] - log_a[j] - 0.2f32 as f64].into_iter().enumerate() {
                let feature = x / (1.0 + (-x).exp());
                summary[unit] += log_l[j].exp() * feature;
                summary[2 + unit] = summary[2 + unit].max(feature);
            }
        }
        let z: Vec<_> = (0..H).map(|j| summary[j] + draft_rows[depth * H + j] as f64).collect();
        for j in 0..H {
            close(nodes[slot * H + j], z[j]);
        }
        let alpha = ((0.2f32 as f64 * z[0] + 0.3f32 as f64).exp().ln_1p()).min(0.8f32 as f64);
        let beta = ((-0.1f32 as f64 * z[1] - 0.3f32 as f64).exp().ln_1p()).min(0.8f32 as f64);
        let scores: Vec<_> = (0..K)
            .map(|j| {
                alpha * l[j]
                    + beta * noise[depth * K + j] as f64
                    + if selected.contains(&j) {
                        0.7f32 as f64
                    } else {
                        0.0
                    }
            })
            .collect();
        let expected = log_softmax(&scores);
        for j in 0..K {
            close(slot_logits[slot * K + j], l[j]);
        }
        for (child, j) in selected.into_iter().enumerate() {
            let entry = slot * C + child;
            close(edges[row * C + child], expected[j]);
            close(f32::from_bits(frontier[FrontierIdx::EdgeLogprobBits as usize * FRONTIER + entry]), expected[j]);
            let path = f32::from_bits(frontier[FrontierIdx::PathLogprobBits as usize * FRONTIER + entry]);
            close(path, parent as f64 + expected[j]);
            let bits = path.to_bits();
            let key = if bits >> 31 == 0 {
                bits ^ (1 << 31)
            } else {
                !bits
            };
            assert_eq!(frontier[FrontierIdx::PathScoreKey as usize * FRONTIER + entry], key);
            assert_eq!(indices[entry], j as u32);
            assert_eq!(real[entry], 1);
            let token = ids[row * K + j] as usize;
            assert_eq!(&inputs[entry * (H + 4)..entry * (H + 4) + H], &table[token * H..(token + 1) * H]);
            let rank = a.iter().filter(|&&value| value > a[j]).count();
            for (number, value) in
                [log_a[j], log_l[j], noise[depth * K + j] as f64, (1.0 + rank as f64).ln()].into_iter().enumerate()
            {
                close(inputs[entry * (H + 4) + H + number], value);
            }
        }
    }
    assert_eq!(&edges[C..2 * C], &[-99.0; C]);
    assert_eq!(&nodes[..H], &[-99.0; H]);
    assert_eq!(&real[..C], &[0; C]);
    for field in [FrontierIdx::TokenId, FrontierIdx::Active, FrontierIdx::ParentSlot, FrontierIdx::Depth] {
        let start = field as usize * FRONTIER;
        assert_eq!(&frontier[start..start + FRONTIER], &frontier_values[start..start + FRONTIER]);
    }
}

#[uzu_test]
fn confidence_head_round_matches_reference() {
    check_round::<Cpu>();
}

#[uzu_test]
fn confidence_head_round_matches_reference_on_gpu() {
    for_each_non_cpu_backend!(|B| {
        check_round::<B>();
    });
}
