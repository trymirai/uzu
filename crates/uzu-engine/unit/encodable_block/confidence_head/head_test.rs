use std::path::Path;

use half::bf16;
use uzu_engine_macros::uzu_test;

use super::*;
use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        gpu_types::weaver::{FrontierIdx, MetadataIdx, TreeIdx},
    },
    config::{
        embedding::AnyEmbeddingConfig,
        speculator::{AnySpeculatorConfig, model::SpeculatorModelConfig},
    },
    safetensors::ParameterLoader,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, for_each_backend},
};

const ROWS: u32 = 15;
const POOL: u32 = 512;
const SLOTS: u32 = 61;
const CHILDREN: u32 = 4;
const MODEL: u32 = 5120;
const VOCAB: u32 = 4096;
const STRIDE: u32 = 15;
const BUDGET: usize = 16;

fn kept(
    slot_entry: &[Option<usize>],
    edges: &[f32],
) -> Vec<usize> {
    let mut reach = vec![0.0f32; edges.len()];
    for entry in 0..edges.len() {
        let parent = slot_entry[entry / CHILDREN as usize];
        reach[entry] = parent.map_or(0.0, |parent| reach[parent]) + edges[entry];
    }
    let mut order = (0..edges.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| reach[right].total_cmp(&reach[left]));
    let mut kept = order[..BUDGET - 1].to_vec();
    kept.sort();
    kept
}

fn report(
    name: &str,
    actual: &[f32],
    expected: &[f32],
) {
    let mut errors =
        actual.iter().zip(expected).map(|(actual, expected)| (actual - expected).abs()).collect::<Vec<_>>();
    errors.sort_by(f32::total_cmp);
    let mean = errors.iter().sum::<f32>() / errors.len() as f32;
    let worst = errors[errors.len() - 1];
    eprintln!("{name}: |edge - reference| mean {mean}, p95 {}, max {worst}", errors[errors.len() * 95 / 100]);
    assert!(mean < 0.02 && worst < 0.1, "{name}: |edge - reference| mean {mean}, max {worst}");
}

async fn check<B: Backend>(
    speculator: &Path,
    fixture: &Path,
) {
    let context = create_context::<B>();
    let config: SpeculatorModelConfig =
        serde_json::from_slice(&std::fs::read(speculator.join("config.json")).unwrap()).unwrap();
    let AnySpeculatorConfig::DFlashSpeculatorConfig(config) = config.speculator_config;
    let weights = ParameterLoader::new(speculator.join("model.safetensors"), &*context).await.unwrap();
    let fixture = ParameterLoader::new(fixture, &*context).await.unwrap();
    let tree = fixture.tree();
    let (embedding, _) = Embedding::new(
        VOCAB,
        MODEL,
        DataType::BF16,
        &serde_json::from_str::<AnyEmbeddingConfig>(
            r#"{"type": "TiedEmbeddingConfig", "input_scale": null, "logit_soft_cap": null, "logit_scale": null}"#,
        )
        .unwrap(),
        &tree,
        &*context,
    )
    .await
    .unwrap();
    let head = ConfidenceHead::<B>::new(
        config.confidence_head_config.as_ref().unwrap(),
        &weights.tree().subtree("speculator").subtree("confidence_head"),
        &embedding,
        STRIDE,
        &*context,
    )
    .await
    .unwrap();

    let (slots, children) = (SLOTS as usize, CHILDREN as usize);
    let candidate_ids = tree.read_slice::<u32>("candidate_ids", &[ROWS, POOL]).await.unwrap();
    let candidate_logits = tree.read_slice::<f32>("candidate_logits", &[ROWS, POOL]).await.unwrap();
    let residuals = tree.read_slice::<bf16>("residuals", &[SLOTS, POOL]).await.unwrap();
    let draft_rows = tree.read_slice::<bf16>("draft_rows", &[ROWS, MODEL]).await.unwrap();
    let depth_seeds = tree.read_slice::<u64>("depth_seeds", &[ROWS]).await.unwrap();
    let slot_token = tree.read_slice::<u32>("slot_token", &[SLOTS]).await.unwrap();
    let slot_parent = tree.read_slice::<i32>("slot_parent", &[SLOTS]).await.unwrap();
    let slot_depth = tree.read_slice::<u32>("slot_depth", &[SLOTS]).await.unwrap();
    let child_token = tree.read_slice::<u32>("child_token", &[SLOTS, CHILDREN]).await.unwrap();
    let child_slot = tree.read_slice::<i32>("child_slot", &[SLOTS, CHILDREN]).await.unwrap();
    let expected_local = tree.read_slice::<f32>("expected_edges_local", &[SLOTS, CHILDREN]).await.unwrap();
    let expected = tree.read_slice::<f32>("expected_edges", &[SLOTS, CHILDREN]).await.unwrap();

    let capacity = slots * children;
    let mut frontier = vec![0u32; FrontierIdx::COUNT * capacity];
    let mut slot_entry = vec![None; slots];
    for entry in 0..capacity {
        let slot = entry / children;
        frontier[FrontierIdx::TokenId as usize * capacity + entry] = child_token[entry];
        frontier[FrontierIdx::ParentSlot as usize * capacity + entry] = slot as u32;
        frontier[FrontierIdx::Depth as usize * capacity + entry] = slot_depth[slot] + 1;
        frontier[FrontierIdx::Active as usize * capacity + entry] = u32::from(child_slot[entry] < 0);
        if child_slot[entry] >= 0 {
            slot_entry[child_slot[entry] as usize] = Some(entry);
        }
    }
    let mut packed_tree = vec![0u32; TreeIdx::COUNT * slots];
    let mut slot_ancestors = vec![0u32; slots * STRIDE as usize];
    let mut metadata = vec![0u32; MetadataIdx::COUNT * slots];
    for slot in 0..slots {
        packed_tree[TreeIdx::TokenId as usize * slots + slot] = slot_token[slot];
        packed_tree[TreeIdx::ParentSlot as usize * slots + slot] = slot_parent[slot] as u32;
        packed_tree[TreeIdx::Depth as usize * slots + slot] = slot_depth[slot];
        packed_tree[TreeIdx::Valid as usize * slots + slot] = 1;
        let mut ancestor = slot_parent[slot];
        while ancestor >= 0 {
            slot_ancestors[slot * STRIDE as usize + slot_depth[ancestor as usize] as usize] = ancestor as u32;
            ancestor = slot_parent[ancestor as usize];
        }
        metadata[MetadataIdx::Depth as usize * slots + slot] = slot_depth[slot];
        metadata[MetadataIdx::AncestorCount as usize * slots + slot] = slot_depth[slot];
        metadata[MetadataIdx::TreeSlot as usize * slots + slot] = slot as u32;
    }
    let draft_hidden =
        std::iter::repeat_n(bf16::ZERO, MODEL as usize).chain(draft_rows.iter().copied()).collect::<Vec<_>>();

    let u32s = |values: &[u32]| create_buffer_with_data::<B, u32>(&context, values);
    let residuals = create_buffer_with_data::<B, bf16>(&context, &residuals);
    let logits = create_buffer_with_data::<B, f32>(&context, &candidate_logits);
    let ids = u32s(&candidate_ids);
    let seeds = create_buffer_with_data::<B, u64>(&context, &depth_seeds);
    let metadata = u32s(&metadata);
    let valid = u32s(&vec![1; slots]);
    let draft_hidden = create_buffer_with_data::<B, bf16>(&context, &draft_hidden);
    let slot_ancestors = u32s(&slot_ancestors);
    let mut packed_tree = u32s(&packed_tree);
    let mut frontier = u32s(&frontier);
    let mut local_frontier = u32s(&vec![0; FrontierIdx::COUNT * capacity]);

    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    let mut step = head
        .encode_step(
            &draft_hidden,
            &logits,
            &ids,
            &seeds,
            ROWS,
            head.params(POOL, CHILDREN, SLOTS, VOCAB),
            &mut command_buffer,
        )
        .unwrap();
    head.encode_round(
        &mut step,
        &residuals,
        &logits,
        &ids,
        &metadata,
        &valid,
        &mut packed_tree,
        &mut frontier,
        SLOTS,
        &mut command_buffer,
    );
    command_buffer.encode_copy(&frontier, &mut local_frontier);
    head.encode_prune(step, &mut packed_tree, &mut frontier, &slot_ancestors, STRIDE, &mut command_buffer).unwrap();
    command_buffer.end_encoding().submit().wait_until_completed().await.unwrap();

    let edges = |frontier| {
        let frontier = buffer_to_vec::<B, u32>(&context, frontier);
        frontier[FrontierIdx::EdgeLogprobBits as usize * capacity..][..capacity]
            .iter()
            .map(|bits| f32::from_bits(*bits))
            .collect::<Vec<_>>()
    };
    let backend = std::any::type_name::<B>();
    let (local, full) = (edges(&local_frontier), edges(&frontier));
    report(&format!("{backend} local"), &local, &expected_local);
    report(&format!("{backend} full"), &full, &expected);
    assert_eq!(
        kept(&slot_entry, &full),
        kept(&slot_entry, &expected),
        "{backend}: the kept nodes differ from the reference's"
    );
}

#[uzu_test]
#[ignore]
fn edges_match_the_reference() {
    let speculator = std::env::var("CONFIDENCE_HEAD_SPECULATOR").expect("CONFIDENCE_HEAD_SPECULATOR is set");
    let fixture = std::env::var("CONFIDENCE_HEAD_FIXTURE").expect("CONFIDENCE_HEAD_FIXTURE is set");
    for_each_backend!(|B| {
        tokio::runtime::Runtime::new().unwrap().block_on(check::<B>(Path::new(&speculator), Path::new(&fixture)));
    });
}
