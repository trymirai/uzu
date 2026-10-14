//! Loading checks run on wasm32, including offsets that cannot fit usize.
use std::{cell::Cell, io};

use js_sys::{Reflect, Uint8Array};
use serde_json::{Value, json};
use uzu_engine::{
    ParameterLoader, ParameterLoaderError, ParameterReadSource,
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
            Context, Kernels, kernel::TensorAddScaleKernel,
        },
        webgpu::{WebGpu, WebGpuError, WebGpuWeightSource, profile_snapshot},
    },
    data_type::DataType,
};
use wasm_bindgen::prelude::*;

type GpuContext = <WebGpu as Backend>::Context;
type Add = <<WebGpu as Backend>::Kernels as Kernels>::TensorAddScaleKernel;

struct SparseSource {
    header: Vec<u8>,
    offset: u64,
    length: u64,
    seen: Cell<u64>,
}
impl ParameterReadSource for SparseSource {
    fn len(&self) -> u64 {
        self.length
    }
    fn read_exact_at(
        &self,
        target: &mut [u8],
        offset: u64,
    ) -> io::Result<()> {
        if offset == self.offset {
            self.seen.set(offset);
            let data = [1.25f32.to_le_bytes(), (-2.5f32).to_le_bytes()].concat();
            if target.len() != data.len() {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            target.copy_from_slice(&data);
        } else {
            let start = usize::try_from(offset).map_err(|_| io::ErrorKind::UnexpectedEof)?;
            let end = start.checked_add(target.len()).ok_or(io::ErrorKind::UnexpectedEof)?;
            target.copy_from_slice(self.header.get(start..end).ok_or(io::ErrorKind::UnexpectedEof)?);
        }
        Ok(())
    }
}

fn sparse_source(
    begin: u64,
    end: u64,
) -> SparseSource {
    let header = serde_json::to_vec(&json!({"weight":{"dtype":"F32","shape":[2],"data_offsets":[begin,end]}})).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    SparseSource {
        offset: (bytes.len() as u64).saturating_add(begin),
        length: (bytes.len() as u64).saturating_add(end),
        header: bytes,
        seen: Cell::new(0),
    }
}

fn stats() -> Result<Value, String> {
    let values: Vec<Value> = serde_json::from_str(&profile_snapshot(false)).map_err(|e| e.to_string())?;
    values.last().cloned().ok_or_else(|| "Missing loading profile".into())
}

fn run(context: &GpuContext) -> Result<Value, String> {
    let mut checks = Vec::new();
    let max_host_size = (isize::MAX as usize) & !63;
    let before = stats()?;
    let limit = match context.create_buffer(max_host_size + 1) {
        Err(WebGpuError::BufferTooLarge {
            limit,
            ..
        }) if limit <= max_host_size => limit,
        _ => return Err("Buffer accepted a size without a valid future CPU mirror".into()),
    };
    if std::alloc::Layout::from_size_align(limit.max(4), 64).is_err() {
        return Err("Reported buffer limit has no valid host layout".into());
    }
    // Merely constructing/dropping a lazy buffer must not allocate its (large)
    // host mirror or GPU storage. This tests the exact boundary with no payload.
    drop(context.create_buffer(limit).map_err(|e| e.to_string())?);
    if !matches!(context.create_buffer(limit + 1), Err(WebGpuError::BufferTooLarge { .. })) {
        return Err("One-byte-over-limit buffer was accepted".into());
    }
    let after = stats()?;
    if before["hostMirrorBytes"] != after["hostMirrorBytes"] || before["buffersCreated"] != after["buffersCreated"] {
        return Err("Boundary validation allocated host/GPU payload memory".into());
    }
    checks.push("per-buffer wasm32 layout boundary rejects oversized CPU mirrors without allocating payload memory");
    let source = sparse_source((1u64 << 32) + 123, (1u64 << 32) + 131);
    let loader = ParameterLoader::<WebGpu>::from_source(&source, context, None).map_err(|e| e.to_string())?;
    let tree = loader.tree();
    let values = tree
        .leaf("weight")
        .and_then(|leaf| leaf.validate(&[2], DataType::F32))
        .and_then(|leaf| leaf.read_slice::<f32>())
        .map_err(|e| e.to_string())?;
    if &*values != [1.25, -2.5] || source.seen.get() <= u32::MAX as u64 {
        return Err("64-bit source offset was truncated".into());
    }
    checks.push("wasm32 reads tensor offsets above 4 GiB without a large allocation");
    for source in [sparse_source(0, 1u64 << 32), sparse_source(u64::MAX - 4, u64::MAX), sparse_source(8, 4)] {
        if ParameterLoader::<WebGpu>::from_source(&source, context, None).is_ok() {
            return Err("Invalid/oversized tensor range accepted".into());
        }
    }
    checks.push("wasm32 rejects tensor-size conversion overflow and invalid absolute offsets");
    let mut short = sparse_source(0, 8);
    short.length -= 1;
    if ParameterLoader::<WebGpu>::from_source(&short, context, None).is_ok() {
        return Err("Truncated source accepted".into());
    }
    checks.push("source length bounds every tensor before initialization");

    let sparse = sparse_source(0, 8);
    let mut bytes = sparse.header;
    bytes.extend(1.25f32.to_le_bytes());
    bytes.extend((-2.5f32).to_le_bytes());
    let source = WebGpuWeightSource::new(Uint8Array::from(bytes.as_slice()).into()).map_err(|e| e.to_string())?;
    let initialize = |context: &_, offset, size| {
        source.initialize_buffer(context, offset, size).map_err(ParameterLoaderError::BackendError)
    };
    let loader =
        ParameterLoader::<WebGpu>::from_source(&source, context, Some(&initialize)).map_err(|e| e.to_string())?;
    let tree = loader.tree();
    let mut input = tree
        .leaf("weight")
        .and_then(|leaf| leaf.validate(&[2], DataType::F32))
        .and_then(|leaf| leaf.read_buffer())
        .map_err(|e| e.to_string())?;
    let mut output = context.create_buffer(8).map_err(|e| e.to_string())?;
    let kernel = Add::new(context, DataType::F32, false).map_err(|e| e.to_string())?;
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    kernel.encode(Some(&input), &input, &mut output, 2, 2, 1.0, &mut command);
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    let before = stats()?;
    if before["hostMirrors"].as_u64().unwrap_or(0) != 0 {
        return Err(format!("Weights allocated CPU shadows: {before}"));
    }
    if before["uploadBytes"].as_u64().unwrap_or(0) != 0 {
        return Err("Weights were uploaded again at first dispatch".into());
    }
    checks.push("parameter GPU upload and first dispatch allocate no CPU mirrors");
    if output.as_slice::<f32>() != [2.5, -5.0] || input.as_slice::<f32>() != [1.25, -2.5] {
        return Err("Lazy CPU mirror did not read GPU weights".into());
    }
    if stats()?["hostMirrorBytes"].as_u64() != Some(16) {
        return Err("CPU mirrors were not allocated on demand".into());
    }
    checks.push("later CPU access reads back a stable retained mirror");
    input.copyin(&[3.0f32, 4.0]);
    let held = input.as_slice::<f32>();
    let pointer = held.as_ptr();
    let mut command = context.create_command_buffer(None, None).map_err(|e| e.to_string())?;
    kernel.encode(Some(&input), &input, &mut output, 2, 2, 1.0, &mut command);
    command.end_encoding().submit().wait_until_completed().map_err(|e| e.to_string())?;
    if held != [3.0, 4.0] || input.as_slice::<f32>().as_ptr() != pointer || output.as_slice::<f32>() != [6.0, 8.0] {
        return Err("CPU mirror ownership/upload regression".into());
    }
    checks.push("CPU mutation uploads correctly without invalidating an outstanding immutable borrow");
    Ok(json!({"checks":checks,"effectiveBufferLimit":limit,"stats":stats()?}))
}

#[wasm_bindgen(js_name = testWebGpuWeightLoading)]
pub async fn test_weights() -> Result<String, JsValue> {
    let global = js_sys::global();
    let key = JsValue::from_str("UZU_WEBGPU_PROFILE");
    let previous = Reflect::get(&global, &key)?;
    Reflect::set(&global, &key, &JsValue::TRUE)?;
    let result = async {
        uzu_engine::backends::webgpu::prepare().await.map_err(|e| e.to_string())?;
        let context = GpuContext::new().map_err(|e| e.to_string())?;
        run(&context)
    }
    .await;
    Reflect::set(&global, &key, &previous)?;
    result.map(|value| value.to_string()).map_err(|e| JsValue::from_str(&e))
}
