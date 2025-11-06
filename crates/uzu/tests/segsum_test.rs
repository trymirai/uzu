#![cfg(any(target_os = "macos", target_os = "ios"))]

use metal::Device;
use mpsgraph::CommandBuffer;
use uzu::{
    Array, DataType, DeviceContext,
    backends::metal::{
        KernelDataType, MTLContext,
        kernel::{
            Cumsum1DArguments, Cumsum1DKernel, SegsumFromCumsumArguments,
            SegsumFromCumsumKernel,
        },
    },
};

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    match MTLContext::new(device, command_queue) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!(
                "Skipping segsum tests: failed to create Metal context: {:?}",
                e
            );
            None
        },
    }
}

#[test]
fn segsum_gpu_invariants_hold() {
    let Some(context) = create_test_context() else {
        return;
    };

    let outer = 3usize;
    let length = 16usize;
    let dtype = DataType::F32;
    let kdt: KernelDataType = dtype.into();

    let mut x = context.array(&[outer, length], dtype);
    let mut s = context.array(&[outer, length], dtype);
    let mut y = context.array(&[outer, length, length], dtype); // zero-initialized

    // Initialize x with a deterministic pattern (no CPU math beyond assignment)
    {
        let xs = x.as_slice_mut::<f32>().unwrap();
        for (i, v) in xs.iter_mut().enumerate() {
            *v = ((i % 7) as f32) / 10.0;
        }
    }

    let cumsum = Cumsum1DKernel::new(&context, kdt).expect("cumsum kernel");
    let segsum =
        SegsumFromCumsumKernel::new(&context, kdt).expect("segsum kernel");

    let command_buffer =
        CommandBuffer::from_command_queue(&context.command_queue);
    let root = command_buffer.root_command_buffer().to_owned();
    let compute = root.new_compute_command_encoder();

    let x_buf = unsafe { x.mtl_buffer() };
    let s_buf = unsafe { s.mtl_buffer() };
    let y_buf = unsafe { y.mtl_buffer() };

    cumsum
        .encode(
            &compute,
            Cumsum1DArguments {
                x: &x_buf,
                s: &s_buf,
                length,
                outer_size: outer,
            },
        )
        .expect("encode cumsum");

    segsum
        .encode(
            &compute,
            SegsumFromCumsumArguments {
                s: &s_buf,
                y: &y_buf,
                length,
                outer_size: outer,
            },
        )
        .expect("encode segsum");

    compute.end_encoding();
    command_buffer.commit();
    root.wait_until_completed();

    // Invariants (checked on CPU without constructing numerical reference):
    // 1) Y[i,i] == 0 and Y[i,j] == 0 for i >= j
    // 2) For j>i: Y[i,j] - Y[i,j-1] == X[j-1]
    let xs = x.as_slice::<f32>().unwrap();
    let ys = y.as_slice::<f32>().unwrap();

    let tol = 1e-4f32;
    for row in 0..outer {
        let x_row_base = row * length;
        let y_row_base = row * length * length;
        for j in 0..length {
            // diagonal must be zero
            let y_diag = ys[y_row_base + j * length + j];
            assert!(
                y_diag.abs() <= tol,
                "diag not zero at row {}, j {}: {}",
                row,
                j,
                y_diag
            );
            for i in 0..length {
                let y_ij = ys[y_row_base + i * length + j];
                if i >= j {
                    assert!(
                        y_ij.abs() <= tol,
                        "upper/diag not zero at row {}, i {}, j {}: {}",
                        row,
                        i,
                        j,
                        y_ij
                    );
                } else {
                    if j > 0 {
                        let y_prev = ys[y_row_base + i * length + (j - 1)];
                        let x_jm1 = xs[x_row_base + (j - 1)];
                        let diff = (y_ij - y_prev) - x_jm1;
                        assert!(
                            diff.abs() <= 1e-3,
                            "increment mismatch at row {}, i {}, j {}: got {}, prev {}, x {} diff {}",
                            row,
                            i,
                            j,
                            y_ij,
                            y_prev,
                            x_jm1,
                            diff
                        );
                    }
                }
            }
        }
    }
}
