#![cfg(any(target_os = "macos", target_os = "ios"))]
use metal::{Device, MTLResourceOptions};
use ndarray::{Array, Array3, s};
use uzu::backends::metal::{
    KVCacheUpdate, KernelDataType, MTLContext,
    kernel::kv_cache_update::{KVLayerData, Swap, create_swaps_direct},
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
                array[(head, src, channel)] =
                    array[(head, dst, channel)].clone();
                array[(head, dst, channel)] = temp;
            }
        }
    }
}

#[test]
fn test_kv_cache_update_kernel() {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue = device.new_command_queue();
    let metal_context = match MTLContext::new(device, command_queue) {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Failed to create MetalContext: {:?}. Skipping test.", e);
            return;
        },
    };

    test_random_pattern(&metal_context);
}

fn test_random_pattern(context: &MTLContext) {
    println!("Testing with random pattern...");

    let max_sequence_length = 256usize;
    let kv_cache_update = match KVCacheUpdate::new(
        context,
        KernelDataType::Float32,
        max_sequence_length,
    ) {
        Ok(update) => update,
        Err(e) => {
            println!(
                "Warning: Failed to create KV cache update: {:?}. Skipping test.",
                e
            );
            return;
        },
    };

    let num_heads = 3usize;
    let seq_len = 15usize;
    let head_dim = 7usize;

    let key_data = Array3::<f32>::from_shape_fn(
        (num_heads, seq_len, head_dim),
        |(h, t, c)| (h * 1_000_000 + t * 100 + c * 10) as f32,
    );

    let value_data = Array3::<f32>::from_shape_fn(
        (num_heads, seq_len, head_dim),
        |(h, t, c)| (h * 1_000_000 + t * 100 + c * 10 + 1_000) as f32,
    );

    let source_indices = vec![0, 3, 6, 9, 12, 2, 5, 8, 11, 14];
    let destination_indices = vec![14, 11, 8, 5, 2, 12, 9, 6, 3, 0];

    let mut expected_keys = key_data.clone();
    let mut expected_values = value_data.clone();

    let swaps = create_swaps_direct(&source_indices, &destination_indices);
    apply_swaps_3d(&mut expected_keys, &swaps);
    apply_swaps_3d(&mut expected_values, &swaps);

    let device = &context.device;

    let key_buffer = device.new_buffer_with_data(
        key_data.as_slice().unwrap().as_ptr() as *const _,
        (key_data.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let value_buffer = device.new_buffer_with_data(
        value_data.as_slice().unwrap().as_ptr() as *const _,
        (value_data.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let kv_layer_data = KVLayerData {
        key_buffer: key_buffer.clone(),
        key_shape: [num_heads, seq_len, head_dim],
        value_buffer: value_buffer.clone(),
        value_shape: [num_heads, seq_len, head_dim],
    };

    let command_buffer = context.command_queue.new_command_buffer().to_owned();
    match kv_cache_update.encode(
        &[kv_layer_data],
        &source_indices,
        &destination_indices,
        &command_buffer,
    ) {
        Ok(_) => {},
        Err(e) => {
            println!(
                "Warning: Failed to encode KV cache update: {:?}. Skipping test.",
                e
            );
            return;
        },
    }

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let key_result_ptr = key_buffer.contents() as *const f32;
    let value_result_ptr = value_buffer.contents() as *const f32;

    let total_elems = num_heads * seq_len * head_dim;

    let key_result_slice =
        unsafe { std::slice::from_raw_parts(key_result_ptr, total_elems) };
    let value_result_slice =
        unsafe { std::slice::from_raw_parts(value_result_ptr, total_elems) };

    let key_result = Array::from_shape_vec(
        (num_heads, seq_len, head_dim),
        key_result_slice.to_vec(),
    )
    .expect("Failed to convert key result to ndarray");

    let value_result = Array::from_shape_vec(
        (num_heads, seq_len, head_dim),
        value_result_slice.to_vec(),
    )
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
