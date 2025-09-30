mod common;

use half::f16;
use uzu::backends::metal::MTLContext;

#[test]
fn test_weight_transform_simple() {
    let E = 2usize;
    let d_model = 4usize;
    let d_ff = 3usize;
    
    let fused: Vec<f16> = (0..E * d_model * d_ff * 2)
        .map(|i| f16::from_f32(i as f32))
        .collect();
    
    eprintln!("Fused weights layout [E={}, d_model={}, 2*d_ff={}]:", E, d_model, d_ff * 2);
    for e in 0..E {
        eprintln!("  Expert {}:", e);
        for d in 0..d_model {
            let start = (e * d_model + d) * d_ff * 2;
            let row: Vec<f32> = fused[start..start + d_ff * 2]
                .iter()
                .map(|v| v.to_f32())
                .collect();
            eprintln!("    d={}: {:?}", d, row);
        }
    }
    
    let device = metal::Device::system_default().unwrap();
    let command_queue = device.new_command_queue();
    let mtl_context = MTLContext::new(device, command_queue).unwrap();
    
    let fused_buf = mtl_context.device.new_buffer_with_data(
        fused.as_ptr() as *const _,
        (fused.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    
    let output_size = (E * d_ff * d_model * 2) as u64;
    let w1_buf = mtl_context.device.new_buffer(
        output_size,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let w3_buf = mtl_context.device.new_buffer(
        output_size,
        metal::MTLResourceOptions::StorageModeShared,
    );
    
    let pipeline = mtl_context
        .compute_pipeline_state("transpose_split_fused_expert_weights_f16", None)
        .unwrap();
    
    let command_queue = mtl_context.device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&fused_buf), 0);
    encoder.set_buffer(1, Some(&w1_buf), 0);
    encoder.set_buffer(2, Some(&w3_buf), 0);
    
    let e_u = E as u32;
    let dm_u = d_model as u32;
    let dff_u = d_ff as u32;
    encoder.set_bytes(3, 4, &e_u as *const u32 as *const _);
    encoder.set_bytes(4, 4, &dm_u as *const u32 as *const _);
    encoder.set_bytes(5, 4, &dff_u as *const u32 as *const _);
    
    let threads_per_tg = metal::MTLSize::new(16, 16, 1);
    let num_tg_x = ((d_model + 15) / 16) as u64;
    let num_tg_y = ((d_ff + 15) / 16) as u64;
    let threadgroups = metal::MTLSize::new(num_tg_x, num_tg_y, E as u64);
    encoder.dispatch_thread_groups(threadgroups, threads_per_tg);
    encoder.end_encoding();
    
    command_buffer.commit();
    command_buffer.wait_until_completed();
    
    let w1_slice: &[f16] = unsafe {
        std::slice::from_raw_parts(
            w1_buf.contents() as *const f16,
            E * d_ff * d_model,
        )
    };
    let w3_slice: &[f16] = unsafe {
        std::slice::from_raw_parts(
            w3_buf.contents() as *const f16,
            E * d_ff * d_model,
        )
    };
    
    eprintln!("\nW1 output [E={}, d_ff={}, d_model={}]:", E, d_ff, d_model);
    for e in 0..E {
        eprintln!("  Expert {}:", e);
        for ff in 0..d_ff {
            let start = (e * d_ff + ff) * d_model;
            let row: Vec<f32> = w1_slice[start..start + d_model]
                .iter()
                .map(|v| v.to_f32())
                .collect();
            eprintln!("    ff={}: {:?}", ff, row);
        }
    }
    
    eprintln!("\nW3 output [E={}, d_ff={}, d_model={}]:", E, d_ff, d_model);
    for e in 0..E {
        eprintln!("  Expert {}:", e);
        for ff in 0..d_ff {
            let start = (e * d_ff + ff) * d_model;
            let row: Vec<f32> = w3_slice[start..start + d_model]
                .iter()
                .map(|v| v.to_f32())
                .collect();
            eprintln!("    ff={}: {:?}", ff, row);
        }
    }
    
    for e in 0..E {
        for ff in 0..d_ff {
            for d in 0..d_model {
                let fused_idx = (e * d_model + d) * d_ff * 2 + ff;
                let expected_w1 = fused[fused_idx].to_f32();
                
                let w1_idx = (e * d_ff + ff) * d_model + d;
                let actual_w1 = w1_slice[w1_idx].to_f32();
                
                assert!(
                    (expected_w1 - actual_w1).abs() < 0.01,
                    "W1 mismatch at e={}, ff={}, d={}: expected {}, got {}",
                    e, ff, d, expected_w1, actual_w1
                );
                
                let fused_idx_gate = (e * d_model + d) * d_ff * 2 + d_ff + ff;
                let expected_w3 = fused[fused_idx_gate].to_f32();
                
                let w3_idx = (e * d_ff + ff) * d_model + d;
                let actual_w3 = w3_slice[w3_idx].to_f32();
                
                assert!(
                    (expected_w3 - actual_w3).abs() < 0.01,
                    "W3 mismatch at e={}, ff={}, d={}: expected {}, got {}",
                    e, ff, d, expected_w3, actual_w3
                );
            }
        }
    }
    
    eprintln!("\n✓ Weight transformation correct!");
}
