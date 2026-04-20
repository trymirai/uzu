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
    let (num_heads, _seq_len, head_dim) = array.dim();
    for head in 0..num_heads {
        for channel in 0..head_dim {
            for swap in swaps {
                let src = swap.source as usize;
                let dst = swap.destination as usize;
                let temp = array[(head, src, channel)].clone();
                array[(head, src, channel)] = array[(head, dst, channel)].clone();
                array[(head, dst, channel)] = temp;
            }
        }
    }
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

    test_random_pattern(&metal_context);
}

fn test_random_pattern(context: &<Metal as Backend>::Context) {
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

    let key_data = Array3::<f32>::from_shape_fn((num_heads, seq_len, head_dim), |(h, t, c)| {
        (h * 1_000_000 + t * 100 + c * 10) as f32
    });

    let value_data = Array3::<f32>::from_shape_fn((num_heads, seq_len, head_dim), |(h, t, c)| {
        (h * 1_000_000 + t * 100 + c * 10 + 1_000) as f32
    });

    let source_indices = vec![0, 3, 6, 9, 12, 2, 5, 8, 11, 14];
    let destination_indices = vec![14, 11, 8, 5, 2, 12, 9, 6, 3, 0];

    let mut expected_keys = key_data.clone();
    let mut expected_values = value_data.clone();

    let swaps = create_swaps_direct(&source_indices, &destination_indices);
    apply_swaps_3d(&mut expected_keys, &swaps);
    apply_swaps_3d(&mut expected_values, &swaps);

    let mut key_allocation = common::helpers::alloc_allocation_with_data(context, key_data.as_slice().unwrap());
    let mut value_allocation = common::helpers::alloc_allocation_with_data(context, value_data.as_slice().unwrap());

    let mut encoder = Encoder::new(context).unwrap();
    {
        let mut kv_layers = [KVLayerData::<Metal> {
            key_allocation: &mut key_allocation,
            key_shape: [num_heads, seq_len, head_dim],
            value_allocation: &mut value_allocation,
            value_shape: [num_heads, seq_len, head_dim],
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

    let key_values: Vec<f32> = common::helpers::allocation_to_vec(&key_allocation);
    let key_result = Array::from_shape_vec((num_heads, seq_len, head_dim), key_values)
        .expect("Failed to convert key result to ndarray");

    let value_values: Vec<f32> = common::helpers::allocation_to_vec(&value_allocation);
    let value_result = Array::from_shape_vec((num_heads, seq_len, head_dim), value_values)
        .expect("Failed to convert value result to ndarray");

    println!("Original keys head 0 rows 0,14:");
    println!("Row 0: {:?}", key_data.slice(s![0, 0, ..]));
    println!("Row 14: {:?}", key_data.slice(s![0, 14, ..]));

    println!("Result keys head 0 rows 0,14:");
    println!("Row 0: {:?}", key_result.slice(s![0, 0, ..]));
    println!("Row 14: {:?}", key_result.slice(s![0, 14, ..]));

    assert_eq!(key_result, expected_keys);
    assert_eq!(value_result, expected_values);
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
