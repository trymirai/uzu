//! Repeat identical shader bodies while changing uniforms and physical aliases.
use serde_json::{Value, json};
use uzu_engine::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, Kernels,
            gpu_types::kv_cache_update::Copy,
            kernel::{KVCacheUpdateKernel, TensorAddScaleKernel},
        },
        webgpu::WebGpu,
    },
    data_type::DataType,
};

type GpuContext = <WebGpu as Backend>::Context;
type Buffer = <WebGpu as Backend>::GlobalBuffer;
type AddScale = <<WebGpu as Backend>::Kernels as Kernels>::TensorAddScaleKernel;
type Update = <<WebGpu as Backend>::Kernels as Kernels>::KVCacheUpdateKernel;

fn buffer(
    context: &GpuContext,
    values: &[f32],
) -> Result<Buffer, String> {
    let mut result = context.create_buffer(values.len() * 4).map_err(|e| e.to_string())?;
    result.copyin(values);
    Ok(result)
}

fn equal(
    label: &str,
    actual: &[f32],
    expected: &[f32],
) -> Result<(), String> {
    if actual != expected {
        let index = actual.iter().zip(expected).position(|(a, b)| a != b);
        return Err(format!("{label}: mismatch at {index:?}, lengths {} / {}", actual.len(), expected.len()));
    }
    Ok(())
}

fn changing_aliases_and_uniforms(context: &GpuContext) -> Result<Value, String> {
    let kernel = AddScale::new(context, DataType::F32, false).map_err(|e| e.to_string())?;
    let layouts = [0, 1, 2, 0, 2, 1, 0];
    let lengths = [3, 257, 513, 17, 259, 9, 515];
    for (step, (&layout, &length)) in layouts.iter().zip(&lengths).enumerate() {
        let input_start = 3 + step;
        let bias_start = 600 + step;
        let output_start = 1100 + step;
        let columns = 2 + step % 4;
        let scale = [0.5, -0.25, 2.0, 1.5][step % 4];
        let input: Vec<f32> = (0..2048).map(|i| ((i + step * 3) % 31) as f32 / 8.0 - 2.0).collect();
        let bias: Vec<f32> = (0..2048).map(|i| ((i + step) % 7) as f32 / 4.0 - 1.0).collect();
        let input_buffer = buffer(context, &input)?;
        let bias_buffer = buffer(context, &bias)?;
        let mut shared = buffer(context, &input)?;
        let mut output = buffer(context, &vec![-777.0; 2048])?;
        let mut expected = if layout == 2 {
            input.clone()
        } else {
            vec![-777.0; 2048]
        };
        for i in 0..length {
            let bias_values = if layout == 0 {
                &bias
            } else {
                &input
            };
            expected[output_start + i] = (input[input_start + i] + bias_values[bias_start + i % columns]) * scale;
        }
        let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
        match layout {
            0 => kernel.encode(
                Some((&input_buffer).subrange(input_start * 4..(input_start + length) * 4)),
                (&bias_buffer).subrange(bias_start * 4..(bias_start + columns) * 4),
                (&mut output).subrange_mut(output_start * 4..(output_start + length) * 4),
                columns as u32,
                length as u32,
                scale,
                &mut command,
            ),
            1 => kernel.encode(
                Some((&shared).subrange(input_start * 4..(input_start + length) * 4)),
                (&shared).subrange(bias_start * 4..(bias_start + columns) * 4),
                (&mut output).subrange_mut(output_start * 4..(output_start + length) * 4),
                columns as u32,
                length as u32,
                scale,
                &mut command,
            ),
            _ => {
                let (read, write) = (&mut shared).split_at(output_start * 4);
                kernel.encode(
                    Some(read.as_ref().subrange(input_start * 4..(input_start + length) * 4)),
                    read.as_ref().subrange(bias_start * 4..(bias_start + columns) * 4),
                    write.subrange_mut(..length * 4),
                    columns as u32,
                    length as u32,
                    scale,
                    &mut command,
                );
            },
        }
        command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
        let actual = if layout == 2 {
            shared.copyout::<f32>()
        } else {
            output.copyout::<f32>()
        };
        equal(&format!("shader cache alias/layout phase {step}"), &actual, &expected)?;
    }
    Ok(json!({"name":"shader cache changing aliases, offsets, uniforms and dispatch sizes","dispatches":layouts.len()}))
}

fn changing_uniform_word_counts(context: &GpuContext) -> Result<Value, String> {
    let kernel = Update::new(context, DataType::F32).map_err(|e| e.to_string())?;
    let lengths = [1, 3, 1, 2, 3, 1];
    // One copy always executes, leaving the WGSL body identical. Extra unused
    // entries change the array shape and move all later uniform/offset slots.
    for (step, &array_length) in lengths.iter().enumerate() {
        let columns = [3, 257, 9][step % 3];
        let row_values = columns * 4;
        let first = Copy {
            source: (step % 4) as u32,
            destination: ((step + 1) % 4) as u32,
        };
        let mut copies = vec![
            Copy {
                source: 2,
                destination: 0
            };
            array_length
        ];
        copies[0] = first;
        let keys: Vec<f32> = (0..row_values).map(|i| i as f32 + step as f32 * 0.5).collect();
        let values: Vec<f32> = keys.iter().map(|value| -value - 17.0).collect();
        let mut expected_keys = keys.clone();
        let mut expected_values = values.clone();
        for i in 0..columns {
            expected_keys[first.destination as usize * columns + i] = keys[first.source as usize * columns + i];
            expected_values[first.destination as usize * columns + i] = values[first.source as usize * columns + i];
        }
        let mut key_buffer = buffer(context, &keys)?;
        let mut value_buffer = buffer(context, &values)?;
        let mut shared = buffer(context, &[keys, values].concat())?;
        let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
        if step % 2 == 0 {
            kernel.encode(&mut key_buffer, &mut value_buffer, &copies, 1, columns as u32, &mut command);
        } else {
            let (key_view, value_view) = (&mut shared).split_at(row_values * 4);
            kernel.encode(key_view, value_view, &copies, 1, columns as u32, &mut command);
        }
        command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
        if step % 2 == 0 {
            equal("shader cache word-count keys", &key_buffer.copyout::<f32>(), &expected_keys)?;
            equal("shader cache word-count values", &value_buffer.copyout::<f32>(), &expected_values)?;
        } else {
            equal(
                "shader cache word-count shared",
                &shared.copyout::<f32>(),
                &[expected_keys, expected_values].concat(),
            )?;
        }
    }
    Ok(json!({"name":"shader cache changing uniform array lengths with unchanged code","dispatches":lengths.len()}))
}

pub(crate) fn run(context: &GpuContext) -> Result<Value, String> {
    Ok(json!({"cases":[changing_aliases_and_uniforms(context)?,changing_uniform_word_counts(context)?]}))
}
