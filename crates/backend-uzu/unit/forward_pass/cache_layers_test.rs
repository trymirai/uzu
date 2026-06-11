#![cfg(metal_backend)]

use std::{mem::size_of, ops::Range};

use proc_macros::uzu_test;

use super::{CacheEntryIndex, CacheLayer, CacheLayers, LayerCacheBinding};
use crate::{
    array::ArrayContextExt,
    backends::{
        common::{Allocation, AllocationType, Backend, Context, Encoder, SparseBuffer},
        metal::Metal,
    },
    data_type::DataType,
    forward_pass::kv_cache_layer::{KVCacheLayer, KVCacheLayerState, KVCacheLayerTrait},
};

fn submit_encoder(encoder: Encoder<Metal>) {
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to submit encoder");
}

fn allocation_from_slice(
    context: &<Metal as Backend>::Context,
    data: &[f32],
) -> Allocation<Metal> {
    context.create_array_from(&[data.len()], data).into_allocation()
}

fn copy_allocation_to_sparse_row(
    context: &<Metal as Backend>::Context,
    allocation: &Allocation<Metal>,
    buffer: &mut <Metal as Backend>::SparseBuffer,
    row_size: usize,
    row_index: usize,
) {
    let row_range = row_byte_range(row_size, row_index);
    let mut encoder = Encoder::<Metal>::new(context).expect("Failed to create encoder");
    encoder.encode_copy(allocation, 0..row_size, buffer, row_range);
    submit_encoder(encoder);
}

fn read_sparse_row(
    context: &<Metal as Backend>::Context,
    buffer: &<Metal as Backend>::SparseBuffer,
    row_size: usize,
    row_index: usize,
) -> Vec<f32> {
    let mut allocation =
        context.create_allocation(row_size, AllocationType::Global).expect("Failed to create allocation");
    let mut encoder = Encoder::<Metal>::new(context).expect("Failed to create encoder");
    encoder.encode_copy(buffer, row_byte_range(row_size, row_index), &mut allocation, 0..row_size);
    submit_encoder(encoder);
    allocation.copyout()
}

fn row_byte_range(
    row_size: usize,
    row_index: usize,
) -> Range<usize> {
    row_index * row_size..(row_index + 1) * row_size
}

fn row_data(
    elements_per_row: usize,
    base_value: f32,
) -> Vec<f32> {
    (0..elements_per_row).map(|element_index| base_value + element_index as f32).collect()
}

fn make_cache_layers(
    context: &<Metal as Backend>::Context,
    state: KVCacheLayerState,
    elements_per_row: usize,
) -> CacheLayers<Metal> {
    let window_length = match state {
        KVCacheLayerState::Windowed {
            window_length,
            ..
        } => window_length,
        KVCacheLayerState::Full {
            ..
        } => panic!("test expects windowed state"),
    };
    let shape = [window_length, elements_per_row, 1];
    let buffer_size = window_length * elements_per_row * size_of::<f32>();
    let layer = KVCacheLayer {
        state,
        keys: context.create_sparse_buffer(buffer_size).expect("Failed to create key buffer"),
        values: context.create_sparse_buffer(buffer_size).expect("Failed to create value buffer"),
        shape,
        data_type: DataType::F32,
    };

    CacheLayers {
        max_suffix_length: 0,
        max_prefix_length: window_length,
        entries: vec![CacheLayer::Transformer(Box::new(layer))].into_boxed_slice(),
        bindings: vec![LayerCacheBinding::Owned {
            entry: CacheEntryIndex {
                index: 0,
            },
        }]
        .into_boxed_slice(),
    }
}

#[uzu_test]
fn test_cache_layers_copy_from_clones_only_live_window_rows() {
    let Some(context) = <Metal as Backend>::Context::new().ok() else {
        return;
    };

    let page_size = context.create_sparse_buffer(1).expect("Failed to create sparse buffer").page_size_bytes();
    let elements_per_row = page_size / size_of::<f32>();
    let row_size = page_size;

    let state = KVCacheLayerState::Windowed {
        ring_offset: 1,
        ring_length: 2,
        window_length: 4,
    };
    let mut source = make_cache_layers(&context, state.clone(), elements_per_row);
    let mut destination = make_cache_layers(
        &context,
        KVCacheLayerState::Windowed {
            ring_offset: 0,
            ring_length: 0,
            window_length: 4,
        },
        elements_per_row,
    );

    let CacheLayer::Transformer(source_layer) = &mut source.entries[0] else {
        panic!("expected transformer layer");
    };
    let source_layer = source_layer
        .as_any_mut()
        .downcast_mut::<KVCacheLayer<Metal, <Metal as Backend>::SparseBuffer>>()
        .expect("expected sparse KV layer");

    source_layer.map_row_range(&context, 1..2).expect("Failed to map source row 1");
    source_layer.map_row_range(&context, 2..3).expect("Failed to map source row 2");

    let row_1_keys = row_data(elements_per_row, 101.0);
    let row_2_keys = row_data(elements_per_row, 102.0);
    let row_1_values = row_data(elements_per_row, 201.0);
    let row_2_values = row_data(elements_per_row, 202.0);

    copy_allocation_to_sparse_row(
        &context,
        &allocation_from_slice(&context, &row_1_keys),
        &mut source_layer.keys,
        row_size,
        1,
    );
    copy_allocation_to_sparse_row(
        &context,
        &allocation_from_slice(&context, &row_2_keys),
        &mut source_layer.keys,
        row_size,
        2,
    );
    copy_allocation_to_sparse_row(
        &context,
        &allocation_from_slice(&context, &row_1_values),
        &mut source_layer.values,
        row_size,
        1,
    );
    copy_allocation_to_sparse_row(
        &context,
        &allocation_from_slice(&context, &row_2_values),
        &mut source_layer.values,
        row_size,
        2,
    );

    destination.copy_from(&source, &context);

    let CacheLayer::Transformer(destination_layer) = &destination.entries[0] else {
        panic!("expected transformer layer");
    };
    let destination_layer = destination_layer
        .as_any()
        .downcast_ref::<KVCacheLayer<Metal, <Metal as Backend>::SparseBuffer>>()
        .expect("expected sparse KV layer");

    assert_eq!(read_sparse_row(&context, &destination_layer.keys, row_size, 1), row_1_keys);
    assert_eq!(read_sparse_row(&context, &destination_layer.keys, row_size, 2), row_2_keys);
    assert_eq!(read_sparse_row(&context, &destination_layer.values, row_size, 1), row_1_values);
    assert_eq!(read_sparse_row(&context, &destination_layer.values, row_size, 2), row_2_values);

    let KVCacheLayerState::Windowed {
        ring_offset,
        ring_length,
        window_length,
    } = destination_layer.state
    else {
        panic!("expected windowed state");
    };
    assert_eq!(ring_offset, 1);
    assert_eq!(ring_length, 2);
    assert_eq!(window_length, 4);
}
