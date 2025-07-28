use std::collections::HashMap;

use metal::{Device, MTLResourceOptions};
use mpsgraph::{
    CommandBuffer, CompilationDescriptor, DataType as MPSDataType,
    Device as MPSDevice, ExecutableExecutionDescriptor, Graph, ShapedType,
    TensorData,
};
use ndarray::{Array2, ArrayView2};
use uzu::{Array, DataType, backends::metal::MetalArray};

fn sequential_f32(len: usize) -> Vec<f32> {
    (0..len).map(|v| v as f32).collect()
}

fn ndarray_matmul(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let a_view = ArrayView2::from_shape((m, k), a).expect("Invalid A shape");
    let b_view = ArrayView2::from_shape((k, n), b).expect("Invalid B shape");
    let c: Array2<f32> = a_view.dot(&b_view);
    let (vec, offset) = c.into_raw_vec_and_offset();
    assert_eq!(offset, Some(0));
    vec
}

#[test]
fn test_row_split_no_copy_simple() {
    let device = match Device::system_default() {
        Some(d) => d,
        None => {
            eprintln!("No Metal device; skipping test");
            return;
        },
    };

    let shape = [4usize, 4usize];
    let num_elems = shape[0] * shape[1];
    let data = sequential_f32(num_elems);

    let buffer_size_bytes = (num_elems * std::mem::size_of::<f32>()) as u64;
    let buffer = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        buffer_size_bytes,
        MTLResourceOptions::StorageModeShared,
    );

    let mut array = unsafe { MetalArray::new(buffer, &shape, DataType::F32) };

    println!("About to call row_split_no_copy");
    let splits = unsafe { array.row_split_no_copy(&[2, 2]) };
    println!("Got {} splits", splits.len());
    assert_eq!(splits.len(), 2);
}

#[test]
fn test_row_split_no_copy_shapes() {
    let device = match Device::system_default() {
        Some(d) => d,
        None => {
            eprintln!("No Metal device; skipping test");
            return;
        },
    };

    let shape = [10usize, 6usize];
    let num_elems = shape[0] * shape[1];
    let data = sequential_f32(num_elems);

    let buffer_size_bytes = (num_elems * std::mem::size_of::<f32>()) as u64;
    let buffer = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        buffer_size_bytes,
        MTLResourceOptions::StorageModeShared,
    );

    let mut array = unsafe { MetalArray::new(buffer, &shape, DataType::F32) };

    let splits = unsafe { array.row_split_no_copy(&[2, 5, 3]) };
    assert_eq!(splits.len(), 3);

    let expected_shapes =
        vec![[2usize, 6usize], [5usize, 6usize], [3usize, 6usize]];
    for (i, (split_array, expected_shape)) in
        splits.iter().zip(expected_shapes.iter()).enumerate()
    {
        let shape = split_array.shape();
        println!(
            "Split {} has shape {:?}, expected {:?}",
            i, shape, expected_shape
        );
        assert_eq!(shape, *expected_shape, "Shape mismatch for split {}", i);
    }

    println!("All row splits have correct shapes!");
}

#[test]
fn test_row_split_no_copy_matmul() {
    let device = match Device::system_default() {
        Some(d) => d,
        None => {
            eprintln!("No Metal device; skipping test");
            return;
        },
    };

    let a_shape = [5usize, 7usize];
    let b_shape = [7usize, 4usize];

    let a_len = a_shape[0] * a_shape[1];
    let b_len = b_shape[0] * b_shape[1];

    let a_data = sequential_f32(a_len);
    let b_data: Vec<f32> = (0..b_len).map(|v| v as f32 * 0.5 + 1.0).collect();

    let b_buffer_size_bytes = (b_len * std::mem::size_of::<f32>()) as u64;
    let b_buffer = device.new_buffer_with_data(
        b_data.as_ptr() as *const _,
        b_buffer_size_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let b_td = TensorData::new_with_mtl_buffer(
        &b_buffer,
        &b_shape,
        MPSDataType::Float32,
        None,
    );

    let baseline =
        ndarray_matmul(a_shape[0], a_shape[1], b_shape[1], &a_data, &b_data);

    let a_buffer_size_bytes = (a_len * std::mem::size_of::<f32>()) as u64;
    let a_buffer = device.new_buffer_with_data(
        a_data.as_ptr() as *const _,
        a_buffer_size_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let mut a_array =
        unsafe { MetalArray::new(a_buffer, &a_shape, DataType::F32) };

    let split_lengths = [2usize, 3usize];
    let mut splits = unsafe { a_array.row_split_no_copy(&split_lengths) };
    assert_eq!(splits.len(), split_lengths.len());

    let mut combined_c = Vec::with_capacity(baseline.len());
    let mut offset = 0usize;
    for (split_array, &rows) in splits.iter_mut().zip(split_lengths.iter()) {
        let slice_start = offset * a_shape[1];
        let slice_end = slice_start + rows * a_shape[1];
        let slice_cpu = &a_data[slice_start..slice_end];
        let expected =
            ndarray_matmul(rows, a_shape[1], b_shape[1], slice_cpu, &b_data);

        let graph = Graph::new();

        let shape_a_isize = [rows as isize, a_shape[1] as isize];
        let a_ph = graph.placeholder(
            Some(&shape_a_isize),
            MPSDataType::Float32,
            Some("A"),
        );

        let shape_b_isize = [a_shape[1] as isize, b_shape[1] as isize];
        let b_ph = graph.placeholder(
            Some(&shape_b_isize),
            MPSDataType::Float32,
            Some("B"),
        );

        let c_tensor =
            graph.matrix_multiplication(&a_ph, &b_ph, Some("MatMul"));

        let mps_device = MPSDevice::with_device(&device);
        let compilation_descriptor = CompilationDescriptor::new();

        let a_shaped_type = ShapedType::new_with_shape_data_type(
            Some(&shape_a_isize),
            MPSDataType::Float32,
        );
        let b_shaped_type = ShapedType::new_with_shape_data_type(
            Some(&shape_b_isize),
            MPSDataType::Float32,
        );

        let mut compile_feeds = HashMap::new();
        compile_feeds.insert(a_ph.as_ref(), a_shaped_type.as_ref());
        compile_feeds.insert(b_ph.as_ref(), b_shaped_type.as_ref());

        let targets = [c_tensor.as_ref()];
        let executable = graph.compile(
            &mps_device,
            &compile_feeds,
            &targets,
            None,
            Some(&compilation_descriptor),
        );

        let result_size = rows * b_shape[1];
        let result_buffer_size =
            (result_size * std::mem::size_of::<f32>()) as u64;
        let result_buffer = device.new_buffer(
            result_buffer_size,
            MTLResourceOptions::StorageModeShared,
        );
        let result_shape_usize = [rows, b_shape[1]];
        let result_td = TensorData::new_with_mtl_buffer(
            &result_buffer,
            &result_shape_usize,
            MPSDataType::Float32,
            None,
        );

        let split_td = unsafe { split_array.to_mps_tensor_data() };

        let inputs = [split_td.as_ref(), b_td.as_ref()];
        let outputs = [result_td.as_ref()];

        let command_queue = device.new_command_queue();
        let command_buffer = CommandBuffer::from_command_buffer(
            &command_queue.new_command_buffer().to_owned(),
        );

        let exec_desc = ExecutableExecutionDescriptor::new();
        exec_desc.set_wait_until_completed(true);

        let _ = executable.encode_to_command_buffer(
            &command_buffer,
            &inputs,
            Some(&outputs),
            Some(&exec_desc),
        );

        command_buffer.commit();
        command_buffer.command_buffer().wait_until_completed();

        let gpu_result = unsafe {
            let ptr = result_buffer.contents() as *const f32;
            std::slice::from_raw_parts(ptr, result_size).to_vec()
        };

        for (i, (&exp, &got)) in
            expected.iter().zip(gpu_result.iter()).enumerate()
        {
            assert!(
                (exp - got).abs() < 1e-4,
                "Mismatch rows slice {} idx {}: expected {}, got {}",
                rows,
                i,
                exp,
                got
            );
        }

        combined_c.extend_from_slice(&expected);

        offset += rows;
    }

    assert_eq!(combined_c.len(), baseline.len());
    for (i, (expected, got)) in
        baseline.iter().zip(combined_c.iter()).enumerate()
    {
        assert!(
            (expected - got).abs() < 1e-6,
            "Mismatch at index {}: expected {}, got {}",
            i,
            expected,
            got
        );
    }

    println!(
        "Matrix multiplication results match for split and unsplit tensors!"
    );
}
