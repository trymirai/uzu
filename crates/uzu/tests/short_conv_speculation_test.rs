#![cfg(target_os = "macos")]

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use uzu::{
    Array,
    DataType,
    DeviceContext,
    backends::metal::{
        KernelDataType, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLContext, MTLDevice, MTLDeviceExt,
        forward_pass::ShortConvLayer,
        kernel::short_conv::{ShortConvKernel, ShortConvTrieArguments},
    },
};

fn approx_eq(a: f32, b: f32) {
    let diff = (a - b).abs();
    assert!(
        diff <= 1e-4,
        "Expected {a} ≈ {b} (diff {diff})"
    );
}

#[test]
fn test_short_conv_trie_kernel_and_commit() {
    let device = <dyn MTLDevice>::system_default().unwrap();
    let queue = device
        .new_command_queue_with_max_command_buffer_count(1024)
        .unwrap();
    let ctx = Rc::new(MTLContext::new(device, queue).unwrap());

    let model_dim = 2usize;
    let kernel_size = 3usize;
    let state_stride = kernel_size - 1;
    let suffix_len = 4usize;
    let in_proj_stride = model_dim * 3;

    let kernel =
        ShortConvKernel::new(&ctx, KernelDataType::Float32).unwrap();

    let mut in_proj = ctx.array(
        &[suffix_len, in_proj_stride],
        DataType::F32,
        String::from("test_in_proj"),
    );
    let mut w = ctx.array(
        &[model_dim, kernel_size],
        DataType::F32,
        String::from("test_w"),
    );
    let out = ctx.array(
        &[suffix_len, model_dim],
        DataType::F32,
        String::from("test_out"),
    );
    let mut parents = ctx.array(
        &[suffix_len],
        DataType::I32,
        String::from("test_parents"),
    );

    let layer = ShortConvLayer {
        conv_state: RefCell::new(ctx.array(
            &[model_dim, state_stride],
            DataType::F32,
            String::from("test_conv_state"),
        )),
        suffix_state: RefCell::new(ctx.array(
            &[suffix_len, model_dim, state_stride],
            DataType::F32,
            String::from("test_suffix_state"),
        )),
        suffix_state_valid_start: Cell::new(0),
        suffix_state_valid_len: Cell::new(0),
    };

    // Base conv_state: channel 0 => [1,2], channel 1 => [3,4]
    {
        let mut cs = layer.conv_state.borrow_mut();
        let cs_slice = cs.as_slice_mut::<f32>().unwrap();
        // shape [model_dim, state_stride] => [2,2]
        cs_slice.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    }

    // Weights:
    // ch0 => [0.1, 0.2, 0.3]
    // ch1 => [0.4, 0.5, 0.6]
    {
        let w_slice = w.as_slice_mut::<f32>().unwrap();
        w_slice.copy_from_slice(&[
            0.1, 0.2, 0.3, //
            0.4, 0.5, 0.6, //
        ]);
    }

    // Parents in DFS order for a small tree:
    // 0(root) -> 1(A) -> 2(AA); and 3(B) is sibling of A under root
    // parent indices are *relative* within this segment.
    {
        let p_slice = parents.as_slice_mut::<i32>().unwrap();
        p_slice.copy_from_slice(&[-1, 0, 1, 0]);
    }

    // in_proj: pre_gate=1, post_gate=1, x_in per token/channel:
    // token0: [10,20], token1: [11,21], token2: [12,22], token3: [13,23]
    {
        let ip = in_proj.as_slice_mut::<f32>().unwrap();
        for t in 0..suffix_len {
            for c in 0..model_dim {
                let base = t * in_proj_stride + c;
                ip[base] = 1.0; // pre
                ip[base + model_dim] = 1.0; // post
                let x = match (t, c) {
                    (0, 0) => 10.0,
                    (0, 1) => 20.0,
                    (1, 0) => 11.0,
                    (1, 1) => 21.0,
                    (2, 0) => 12.0,
                    (2, 1) => 22.0,
                    (3, 0) => 13.0,
                    (3, 1) => 23.0,
                    _ => unreachable!(),
                };
                ip[base + 2 * model_dim] = x;
            }
        }
    }

    // Run trie kernel.
    let cmd = ctx
        .command_queue
        .command_buffer()
        .expect("Failed to create command buffer")
        .to_owned();
    let encoder = cmd
        .new_compute_command_encoder()
        .expect("Failed to create compute encoder");

    let in_proj_buf = in_proj.mtl_buffer_cloned();
    let w_buf = w.mtl_buffer_cloned();
    let out_buf = out.mtl_buffer_cloned();
    let parents_buf = parents.mtl_buffer_cloned();
    let base_state_buf = layer.conv_state.borrow().mtl_buffer_cloned();
    let suffix_state_buf = layer.suffix_state.borrow().mtl_buffer_cloned();

    kernel
        .encode_trie(
            &encoder,
            ShortConvTrieArguments {
                in_proj: &in_proj_buf,
                in_proj_offset: 0,
                w: &w_buf,
                b: None,
                base_state: &base_state_buf,
                base_state_offset: 0,
                parents: &parents_buf,
                parents_offset: 0,
                out: &out_buf,
                out_offset: 0,
                suffix_state: &suffix_state_buf,
                suffix_state_offset: 0,
                suffix_len,
                kernel_size: kernel_size as i32,
                in_proj_stride,
                state_stride,
                model_dim,
            },
        )
        .unwrap();

    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Validate outputs.
    let out_slice = out.as_slice::<f32>().unwrap();
    // channel0 expected: [3.5, 5.5, 6.8, 6.1]
    approx_eq(out_slice[0 * model_dim + 0], 3.5);
    approx_eq(out_slice[1 * model_dim + 0], 5.5);
    approx_eq(out_slice[2 * model_dim + 0], 6.8);
    approx_eq(out_slice[3 * model_dim + 0], 6.1);
    // channel1 expected: [15.2, 24.2, 31.7, 25.4]
    approx_eq(out_slice[0 * model_dim + 1], 15.2);
    approx_eq(out_slice[1 * model_dim + 1], 24.2);
    approx_eq(out_slice[2 * model_dim + 1], 31.7);
    approx_eq(out_slice[3 * model_dim + 1], 25.4);

    // Validate suffix_state post-states.
    let ss = layer.suffix_state.borrow();
    let ss_slice = ss.as_slice::<f32>().unwrap();
    let idx = |t: usize, c: usize, tap: usize| -> usize {
        (t * model_dim + c) * state_stride + tap
    };

    // root state: ch0 [2,10], ch1 [4,20]
    approx_eq(ss_slice[idx(0, 0, 0)], 2.0);
    approx_eq(ss_slice[idx(0, 0, 1)], 10.0);
    approx_eq(ss_slice[idx(0, 1, 0)], 4.0);
    approx_eq(ss_slice[idx(0, 1, 1)], 20.0);

    // AA state (token2): ch0 [11,12], ch1 [21,22]
    approx_eq(ss_slice[idx(2, 0, 0)], 11.0);
    approx_eq(ss_slice[idx(2, 0, 1)], 12.0);
    approx_eq(ss_slice[idx(2, 1, 0)], 21.0);
    approx_eq(ss_slice[idx(2, 1, 1)], 22.0);

    // Commit checks (simulate acceptance).
    layer.set_suffix_state_valid_range(0, suffix_len);

    // Empty accepted_suffix_indices => commit root state (index 0)
    layer.commit_from_suffix_state_if_valid(0);
    {
        let cs = layer.conv_state.borrow();
        let cs_slice = cs.as_slice::<f32>().unwrap();
        approx_eq(cs_slice[0], 2.0);
        approx_eq(cs_slice[1], 10.0);
        approx_eq(cs_slice[2], 4.0);
        approx_eq(cs_slice[3], 20.0);
    }

    // Commit deepest main-branch node (index 2)
    layer.commit_from_suffix_state_if_valid(2);
    {
        let cs = layer.conv_state.borrow();
        let cs_slice = cs.as_slice::<f32>().unwrap();
        approx_eq(cs_slice[0], 11.0);
        approx_eq(cs_slice[1], 12.0);
        approx_eq(cs_slice[2], 21.0);
        approx_eq(cs_slice[3], 22.0);
    }

    // Valid-range guard: only [2..4) is valid; committing 1 must not change conv_state.
    layer.set_suffix_state_valid_range(2, 2);
    {
        let mut cs = layer.conv_state.borrow_mut();
        let cs_slice = cs.as_slice_mut::<f32>().unwrap();
        cs_slice.copy_from_slice(&[100.0, 101.0, 102.0, 103.0]);
    }
    layer.commit_from_suffix_state_if_valid(1);
    {
        let cs = layer.conv_state.borrow();
        let cs_slice = cs.as_slice::<f32>().unwrap();
        approx_eq(cs_slice[0], 100.0);
        approx_eq(cs_slice[1], 101.0);
        approx_eq(cs_slice[2], 102.0);
        approx_eq(cs_slice[3], 103.0);
    }
}

