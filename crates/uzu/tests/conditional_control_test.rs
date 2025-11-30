#![cfg(target_os = "macos")]

use std::time::{Duration, Instant};

use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLCompareFunction, MTLResourceOptions, MTLResourceUsage, MTLSize,
};
use uzu::backends::metal::metal_extensions::ComputeEncoderConditional;

#[test]
fn test_conditional_loop_predicate_update_api() {
    // Setup device and queue
    let device = Device::system_default().expect("No Metal device found");
    let command_queue: CommandQueue = device.new_command_queue();

    // Kernel that increments predicate[0] by 1
    let src = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void predicate_update(device uint* predicate [[ buffer(0) ]]) {
            predicate[0] += 1u;
        }
    "#;

    // Build pipeline
    let options = CompileOptions::new();
    let library: Library = device
        .new_library_with_source(src, &options)
        .expect("Failed to compile kernel source");
    let function = library
        .get_function("predicate_update", None)
        .expect("Failed to find kernel function");
    let pipeline: ComputePipelineState = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Failed to create pipeline state");

    // Predicate buffer initial value 0
    let init_predicate: u32 = 0;
    let predicate_buffer = device.new_buffer_with_data(
        &init_predicate as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Sanity check: single kernel dispatch increments to 1
    {
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&predicate_buffer), 0);
        let tg = MTLSize::new(1, 1, 1);
        encoder.dispatch_thread_groups(tg, tg);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    unsafe {
        let p = predicate_buffer.contents() as *mut u32;
        assert_eq!(*p, 1u32, "Kernel failed to update predicate");
        // Reset to 0 for loop test
        *p = 0u32;
    }

    // Loop: while (predicate <= 128) increment by 1
    let reference_value: u32 = 128;
    {
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Ensure visibility for conditional logic
        encoder.use_resource(
            &predicate_buffer,
            MTLResourceUsage::Read | MTLResourceUsage::Write,
        );

        // High-level while semantics; implementation inverts comparison for msg_send
        encoder.while_loop(
            &predicate_buffer,
            0,
            MTLCompareFunction::LessEqual,
            reference_value,
            || {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&predicate_buffer), 0);
                let tg = MTLSize::new(1, 1, 1);
                encoder.dispatch_thread_groups(tg, tg);
            },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    unsafe {
        let p = predicate_buffer.contents() as *const u32;
        assert_eq!(*p, 128u32, "Loop executed until predicate reached 128");
    }
}

#[test]
fn test_conditional_if_execution_api() {
    // Setup device and queue
    let device = Device::system_default().expect("No Metal device found");
    let command_queue: CommandQueue = device.new_command_queue();

    // Two trivial kernels: write 1 or 2 into data[0]
    let src = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void set_value_1(device uint* data [[ buffer(0) ]]) { data[0] = 1u; }
        kernel void set_value_2(device uint* data [[ buffer(0) ]]) { data[0] = 2u; }
    "#;

    let options = CompileOptions::new();
    let library: Library = device
        .new_library_with_source(src, &options)
        .expect("Failed to compile kernel source");

    let f1 = library
        .get_function("set_value_1", None)
        .expect("Failed to find set_value_1");
    let f2 = library
        .get_function("set_value_2", None)
        .expect("Failed to find set_value_2");

    let p1 = device
        .new_compute_pipeline_state_with_function(&f1)
        .expect("Failed to create pipeline state 1");
    let p2 = device
        .new_compute_pipeline_state_with_function(&f2)
        .expect("Failed to create pipeline state 2");

    // Predicate buffer initialized to 10, result buffer to 0
    let predicate_value: u32 = 10;
    let predicate_buffer = device.new_buffer_with_data(
        &predicate_value as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let result_value: u32 = 0;
    let result_buffer = device.new_buffer_with_data(
        &result_value as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    // Ensure visibility for conditional logic
    encoder.use_resource(&predicate_buffer, MTLResourceUsage::Read);

    // if (predicate == 0) then write 1 else write 2
    encoder.condition(
        &predicate_buffer,
        0,
        MTLCompareFunction::Equal,
        0,
        || {
            encoder.set_compute_pipeline_state(&p1);
            encoder.set_buffer(0, Some(&result_buffer), 0);
            let tg = MTLSize::new(1, 1, 1);
            encoder.dispatch_thread_groups(tg, tg);
        },
        Some(|| {
            encoder.set_compute_pipeline_state(&p2);
            encoder.set_buffer(0, Some(&result_buffer), 0);
            let tg = MTLSize::new(1, 1, 1);
            encoder.dispatch_thread_groups(tg, tg);
        }),
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    unsafe {
        let r = result_buffer.contents() as *const u32;
        assert_eq!(
            *r, 2u32,
            "Expected else branch (value 2) since predicate was 10"
        );
    }
}

#[test]
fn test_conditional_loop_external_predicate_update_api() {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue: CommandQueue = device.new_command_queue();

    let src = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void bump_counter(device uint* counter [[ buffer(0) ]]) {
            counter[0] += 1u;
        }
    "#;

    let options = CompileOptions::new();
    let library: Library = device
        .new_library_with_source(src, &options)
        .expect("Failed to compile kernel source");
    let function = library
        .get_function("bump_counter", None)
        .expect("Failed to find bump_counter");
    let pipeline: ComputePipelineState = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Failed to create pipeline state");

    let predicate_value: u32 = 0;
    let predicate_buffer = device.new_buffer_with_data(
        &predicate_value as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let counter_value: u32 = 0;
    let counter_buffer = device.new_buffer_with_data(
        &counter_value as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let pred_ptr = predicate_buffer.contents() as *mut u32;

    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.use_resource(&predicate_buffer, MTLResourceUsage::Read);
    encoder.use_resource(&counter_buffer, MTLResourceUsage::Write);

    // while (predicate <= 1) { bump counter }.
    // Kernel does NOT modify predicate; CPU will flip predicate later.
    encoder.while_loop(
        &predicate_buffer,
        0,
        MTLCompareFunction::LessEqual,
        1,
        || {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&counter_buffer), 0);
            let tg = MTLSize::new(1, 1, 1);
            encoder.dispatch_thread_groups(tg, tg);
        },
    );

    encoder.end_encoding();

    // Start GPU work
    command_buffer.commit();

    // Flip predicate from CPU after a short delay; GPU should stop looping.
    std::thread::sleep(Duration::from_secs(1));
    unsafe {
        *pred_ptr = 1u32;
    }

    let start = Instant::now();
    command_buffer.wait_until_completed();
    let elapsed = start.elapsed();

    unsafe {
        let progressed = *(counter_buffer.contents() as *const u32);
        assert!(progressed > 0, "Counter should have incremented in the loop");
    }

    // Not asserting on exact timing, but we expect completion shortly after the flip.
    eprintln!(
        "Loop terminated after CPU predicate update, wait: {:.2?}",
        elapsed
    );
}

#[test]
fn test_conditional_if_greater_without_else_api() {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue: CommandQueue = device.new_command_queue();

    let src = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void write_value_3(device uint* data [[ buffer(0) ]]) { data[0] = 3u; }
    "#;

    let options = CompileOptions::new();
    let library: Library = device
        .new_library_with_source(src, &options)
        .expect("Failed to compile kernel source");
    let func = library
        .get_function("write_value_3", None)
        .expect("Failed to find write_value_3");
    let pipeline: ComputePipelineState = device
        .new_compute_pipeline_state_with_function(&func)
        .expect("Failed to create pipeline state");

    let predicate_value: u32 = 7;
    let predicate_buffer = device.new_buffer_with_data(
        &predicate_value as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let result_value: u32 = 0;
    let result_buffer = device.new_buffer_with_data(
        &result_value as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.use_resource(&predicate_buffer, MTLResourceUsage::Read);

    encoder.condition(
        &predicate_buffer,
        0,
        MTLCompareFunction::Greater,
        3,
        || {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&result_buffer), 0);
            let tg = MTLSize::new(1, 1, 1);
            encoder.dispatch_thread_groups(tg, tg);
        },
        None::<fn()>,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    unsafe {
        let r = result_buffer.contents() as *const u32;
        assert_eq!(*r, 3u32, "Then-branch should execute and write 3");
    }
}

#[test]
fn test_conditional_if_offset_equal_api() {
    let device = Device::system_default().expect("No Metal device found");
    let command_queue: CommandQueue = device.new_command_queue();

    let src = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void write_value_4(device uint* data [[ buffer(0) ]]) { data[0] = 4u; }
        kernel void write_value_5(device uint* data [[ buffer(0) ]]) { data[0] = 5u; }
    "#;

    let options = CompileOptions::new();
    let library: Library = device
        .new_library_with_source(src, &options)
        .expect("Failed to compile kernel source");

    let f4 = library
        .get_function("write_value_4", None)
        .expect("Failed to find write_value_4");
    let f5 = library
        .get_function("write_value_5", None)
        .expect("Failed to find write_value_5");

    let p4 = device
        .new_compute_pipeline_state_with_function(&f4)
        .expect("Failed to create pipeline state 4");
    let p5 = device
        .new_compute_pipeline_state_with_function(&f5)
        .expect("Failed to create pipeline state 5");

    let predicate_initial: [u32; 2] = [0, 10];
    let predicate_buffer = device.new_buffer_with_data(
        predicate_initial.as_ptr() as *const _,
        (predicate_initial.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let result_value: u32 = 0;
    let result_buffer = device.new_buffer_with_data(
        &result_value as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.use_resource(&predicate_buffer, MTLResourceUsage::Read);

    encoder.condition(
        &predicate_buffer,
        4,
        MTLCompareFunction::Equal,
        10,
        || {
            encoder.set_compute_pipeline_state(&p4);
            encoder.set_buffer(0, Some(&result_buffer), 0);
            let tg = MTLSize::new(1, 1, 1);
            encoder.dispatch_thread_groups(tg, tg);
        },
        Some(|| {
            encoder.set_compute_pipeline_state(&p5);
            encoder.set_buffer(0, Some(&result_buffer), 0);
            let tg = MTLSize::new(1, 1, 1);
            encoder.dispatch_thread_groups(tg, tg);
        }),
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    unsafe {
        let r = result_buffer.contents() as *const u32;
        assert_eq!(
            *r, 4u32,
            "Then-branch should execute (offset read equals 10)"
        );
    }
}
