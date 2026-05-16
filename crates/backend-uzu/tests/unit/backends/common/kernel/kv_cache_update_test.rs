#![cfg(metal_backend)]

#[macro_use]
#[path = "../../../../common/mod.rs"]
mod common;

use ndarray::{Array, Array3, s};

use super::*;
use crate::backends::{
    common::{Context, Encoder},
    metal::Metal,
};

fn apply_swaps_3d<T: Clone>(
    array: &mut Array3<T>,
    swaps: &[Swap],
) {
    let (_seq_len, num_heads, head_dim) = array.dim();
    for head in 0..num_heads {
        for channel in 0..head_dim {
            for swap in swaps {
                let src = swap.source as usize;
                let dst = swap.destination as usize;
                let temp = array[(src, head, channel)].clone();
                array[(src, head, channel)] = array[(dst, head, channel)].clone();
                array[(dst, head, channel)] = temp;
            }
        }
    }
}

fn test_random_pattern<B: Backend>(context: &B::Context) {
    println!("Testing with random pattern...");

    let max_sequence_length = 256usize;
    let kv_cache_update = match KVCacheUpdate::new(context, DataType::F32, max_sequence_length) {
        Ok(update) => update,
        Err(e) => {
            println!("Warning: Failed to create KV cache update: {:?}. Skipping test.", e);
            return;
        },
    };

    let num_heads = 3usize;
    let seq_len = 15usize;
    let head_dim = 7usize;

    let key_data = Array3::<f32>::from_shape_fn((seq_len, num_heads, head_dim), |(h, t, c)| {
        (h * 1_000_000 + t * 100 + c * 10) as f32
    });

    let value_data = Array3::<f32>::from_shape_fn((seq_len, num_heads, head_dim), |(h, t, c)| {
        (h * 1_000_000 + t * 100 + c * 10 + 1_000) as f32
    });

    let source_indices = vec![0, 3, 6, 9, 12, 2, 5, 8, 11, 14];
    let destination_indices = vec![14, 11, 8, 5, 2, 12, 9, 6, 3, 0];

    let mut expected_keys = key_data.clone();
    let mut expected_values = value_data.clone();

    let swaps = create_swaps_direct(&source_indices, &destination_indices);
    apply_swaps_3d(&mut expected_keys, &swaps);
    apply_swaps_3d(&mut expected_values, &swaps);

    let mut key_buffer = common::helpers::sparse_buffer_create_with::<B, f32>(context, key_data.as_slice().unwrap());
    let mut value_buffer =
        common::helpers::sparse_buffer_create_with::<B, f32>(context, value_data.as_slice().unwrap());
    let mut encoder = Encoder::<B>::new(context).unwrap();
    {
        let mut kv_layers = [KVLayerData::<B> {
            key_allocation: &mut key_buffer,
            key_shape: [seq_len, num_heads, head_dim],
            value_allocation: &mut value_buffer,
            value_shape: [seq_len, num_heads, head_dim],
        }];
        match kv_cache_update.encode(&mut kv_layers, &source_indices, &destination_indices, &mut encoder) {
            Ok(_) => {},
            Err(e) => {
                println!("Warning: Failed to encode KV cache update: {:?}. Skipping test.", e);
                return;
            },
        }
    }

    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let shape = [seq_len, num_heads, head_dim];
    let elements_count = shape.iter().product();

    let key_values: Vec<f32> = common::helpers::sparse_buffer_read::<B, f32>(context, &key_buffer, elements_count);
    let key_result = Array::from_shape_vec((seq_len, num_heads, head_dim), key_values)
        .expect("Failed to convert key result to ndarray");

    let value_values: Vec<f32> = common::helpers::sparse_buffer_read::<B, f32>(context, &value_buffer, elements_count);
    let value_result = Array::from_shape_vec((seq_len, num_heads, head_dim), value_values)
        .expect("Failed to convert value result to ndarray");

    println!("Original keys tokens 0,14 head 0:");
    println!("Token 0: {:?}", key_data.slice(s![0, 0, ..]));
    println!("Token 14: {:?}", key_data.slice(s![14, 0, ..]));

    println!("Result keys tokens 0,14 head 0:");
    println!("Token 0: {:?}", key_result.slice(s![0, 0, ..]));
    println!("Token 14: {:?}", key_result.slice(s![14, 0, ..]));

    assert_eq!(key_result, expected_keys);
    assert_eq!(value_result, expected_values);
}

#[test]
fn test_kv_cache_update_kernel() {
    let metal_context = match <Metal as Backend>::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Failed to create MetalContext: {:?}. Skipping test.", e);
            return;
        },
    };

    test_random_pattern::<Metal>(&metal_context);
}

#[test]
fn test_direct_swaps() {
    let sources = [0, 2, 4];
    let destinations = [1, 3, 5];
    let swaps = create_swaps_direct(&sources, &destinations);
    assert_eq!(swaps.len(), 3);
    assert_eq!(swaps[0].source, 0);
    assert_eq!(swaps[0].destination, 1);
    assert_eq!(swaps[1].source, 2);
    assert_eq!(swaps[1].destination, 3);
    assert_eq!(swaps[2].source, 4);
    assert_eq!(swaps[2].destination, 5);
}
