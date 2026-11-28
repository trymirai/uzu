use std::{panic::AssertUnwindSafe, path::PathBuf, sync::Arc};

use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::cpu::Cpu,
    data_type::DataType,
    encodable_block::embedding_table_disk::{EmbeddingTableDisk, EmbeddingTableDiskError, SharedFiles, lookup},
    safetensors::{ParameterLoader, ParameterLoaderError},
    tests::helpers::create_context,
};

const NUM_ROWS: u32 = 37;
const ROW_DIM: u32 = 6;
const ROW_SIZE: usize = ROW_DIM as usize * 2;

/// Row `index` holds `index * 100 + column` so any misread is attributable.
fn row(index: u32) -> Vec<u8> {
    (0..ROW_DIM).flat_map(|column| bf16::from_f32((index * 100 + column) as f32).to_le_bytes()).collect()
}

fn write_table(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("uzu-{name}-{}.safetensors", std::process::id()));
    let bytes = (0..NUM_ROWS).flat_map(row).collect::<Vec<u8>>();
    let header = serde_json::to_vec(&serde_json::json!({
        "ngram_embedding.weights": {"dtype": "BF16", "shape": [NUM_ROWS, ROW_DIM], "data_offsets": [0, bytes.len()]}
    }))
    .expect("safetensors header");
    std::fs::write(&path, [(header.len() as u64).to_le_bytes().as_slice(), &header, &bytes].concat())
        .expect("write safetensors");
    path
}

/// Loads the table and drops the loader, so what comes back only has the
/// duplicated descriptor to read from.
fn load_table(name: &str) -> (EmbeddingTableDisk, PathBuf) {
    let path = write_table(name);
    let context = create_context::<Cpu>();
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let table = runtime.block_on(async {
        let loader = ParameterLoader::<Cpu>::new(&path, &context).await.expect("parameter loader");
        let table = EmbeddingTableDisk::load(
            NUM_ROWS,
            ROW_DIM,
            DataType::BF16,
            &loader.tree().subtree("ngram_embedding"),
            &mut SharedFiles::default(),
        )
        .await
        .expect("load table");
        loader.assert_all_tensors_read().expect("claim marks the tensor as read");
        table
    });
    (table, path)
}

#[uzu_test]
fn load_shares_file_handles() {
    let path = write_table("ngram-table-shared");
    let context = create_context::<Cpu>();
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let (first, second, separate) = runtime.block_on(async {
        let loader = ParameterLoader::<Cpu>::new(&path, &context).await.expect("parameter loader");
        let tree = loader.tree().subtree("ngram_embedding");
        let load = async |files: &mut SharedFiles| {
            EmbeddingTableDisk::load(NUM_ROWS, ROW_DIM, DataType::BF16, &tree, files).await.expect("load table")
        };
        let mut files = SharedFiles::default();
        let first = load(&mut files).await;
        let second = load(&mut files).await;
        let separate = load(&mut SharedFiles::default()).await;
        (first, second, separate)
    });
    std::fs::remove_file(&path).expect("remove safetensors");

    assert!(Arc::ptr_eq(&first.file, &second.file), "tables loaded with the same `SharedFiles` share a handle");
    assert!(!Arc::ptr_eq(&first.file, &separate.file), "tables loaded with different `SharedFiles` don't");
}

#[uzu_test]
fn lookup_reads_rows_after_loader_and_file_are_gone() {
    let (table, path) = load_table("ngram-table");
    std::fs::remove_file(&path).expect("remove safetensors");

    let ids = [0, NUM_ROWS - 1, 5, 5, 17];
    let rows = lookup(&[(&table, ids)]).expect("lookup").pop().unwrap();
    assert_eq!(rows.len(), ids.len() * ROW_SIZE);
    for (&id, actual) in ids.iter().zip(rows.as_chunks::<ROW_SIZE>().0) {
        assert_eq!(actual, row(id).as_slice(), "row {id}");
    }
}

#[uzu_test]
fn lookup_keeps_lookups_in_order() {
    let (table, path) = load_table("ngram-table-many");
    std::fs::remove_file(&path).expect("remove safetensors");

    let lookups = [(&table, vec![9, 2, 9]), (&table, vec![]), (&table, vec![2, 30])];
    let rows = lookup(&lookups).expect("lookup");
    assert_eq!(rows.len(), lookups.len());
    for ((_, ids), rows) in lookups.iter().zip(&rows) {
        assert_eq!(*rows, ids.iter().flat_map(|&id| row(id)).collect::<Vec<_>>());
    }
}

#[uzu_test]
fn lookup_rejects_out_of_range_id() {
    let (table, path) = load_table("ngram-table-range");
    std::fs::remove_file(&path).expect("remove safetensors");

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| lookup(&[(&table, [NUM_ROWS])])));
    assert!(result.is_err(), "id equal to the row count must not be read");
}

#[uzu_test]
fn load_rejects_wrong_shape_without_claiming() {
    let path = write_table("ngram-table-shape");
    let context = create_context::<Cpu>();
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async {
        let loader = ParameterLoader::<Cpu>::new(&path, &context).await.expect("parameter loader");
        let tree = loader.tree().subtree("ngram_embedding");
        let result =
            EmbeddingTableDisk::load(NUM_ROWS, ROW_DIM + 1, DataType::BF16, &tree, &mut SharedFiles::default()).await;
        assert!(matches!(
            result,
            Err(EmbeddingTableDiskError::ParameterLoaderError(ParameterLoaderError::InvalidTensor { .. }))
        ));
        assert!(
            matches!(loader.assert_all_tensors_read(), Err(ParameterLoaderError::UnreadTensors(_))),
            "a failed claim must leave the tensor unread"
        );
    });
    std::fs::remove_file(&path).expect("remove safetensors");
}
