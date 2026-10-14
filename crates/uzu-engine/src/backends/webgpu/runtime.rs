use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
};

use parking_lot::Mutex;
use serde::Deserialize;
use wasm_bindgen::prelude::*;

use super::WebGpuError;

#[wasm_bindgen(module = "/src/backends/webgpu/bridge.js")]
extern "C" {
    #[wasm_bindgen(catch, js_name = weightSourceSize)]
    pub(super) fn weight_source_size(source: &JsValue) -> Result<f64, JsValue>;
    #[wasm_bindgen(catch, js_name = checkLoadCancellation)]
    pub(super) fn check_load_cancellation(flag: &JsValue) -> Result<(), JsValue>;
    #[wasm_bindgen(catch, js_name = readWeightSource)]
    pub(super) fn read_weight_source(
        source: &JsValue,
        offset: f64,
        destination: &mut [u8],
        cancellation: &JsValue,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(catch, js_name = initializeBufferFromSource)]
    fn initialize_buffer_from_source(
        context: u32,
        buffer: u32,
        source: &JsValue,
        offset: f64,
        size: usize,
        cancellation: &JsValue,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(js_name = recordHostAllocation)]
    fn record_host_allocation(
        context: u32,
        size: usize,
    );
    #[wasm_bindgen(js_name = profileNow)]
    fn profile_now() -> f64;
    #[wasm_bindgen(js_name = profileElapsed)]
    fn profile_elapsed(
        context: u32,
        stage: &str,
        started: f64,
    );
    #[wasm_bindgen(js_name = profileSnapshot)]
    pub(super) fn profile_snapshot(reset: bool) -> String;
    #[wasm_bindgen(catch, js_name = prepareContext)]
    pub(super) async fn prepare_context() -> Result<(), JsValue>;
    #[wasm_bindgen(catch, js_name = createContext)]
    fn create_context() -> Result<String, JsValue>;
    #[wasm_bindgen(js_name = destroyContext)]
    fn destroy_context(context: u32);
    #[wasm_bindgen(catch, js_name = createBuffer)]
    fn create_buffer(
        context: u32,
        buffer: u32,
        size: usize,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(js_name = destroyBuffer)]
    fn destroy_buffer(
        context: u32,
        buffer: u32,
    );
    #[wasm_bindgen(catch, js_name = uploadBuffer)]
    fn upload_buffer(
        context: u32,
        buffer: u32,
        bytes: &[u8],
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(catch, js_name = readBuffer)]
    fn read_buffer(
        context: u32,
        buffer: u32,
        bytes: &mut [u8],
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(catch, js_name = registerPipeline)]
    fn register_pipeline(
        context: u32,
        pipeline: u32,
        source: &str,
    ) -> Result<(), JsValue>;
    #[wasm_bindgen(catch, js_name = submit)]
    fn submit_commands(
        context: u32,
        commands: &[u32],
        written_buffers: &[u32],
    ) -> Result<u32, JsValue>;
    #[wasm_bindgen(catch, js_name = wait)]
    fn wait_commands(
        context: u32,
        submission: u32,
    ) -> Result<f64, JsValue>;
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DeviceInfo {
    id: u32,
    name: String,
    max_buffer_size: usize,
    max_storage_buffer_binding_size: usize,
    #[serde(default)]
    profiling: bool,
    #[serde(default)]
    supports_subgroups: bool,
    #[serde(default)]
    subgroup_min_size: u32,
    #[serde(default)]
    subgroup_max_size: u32,
    #[serde(default)]
    subgroup_id_supported: bool,
}

pub(super) struct Runtime {
    pub id: u32,
    pub name: String,
    pub max_buffer_size: usize,
    next_buffer: AtomicU32,
    pipelines: Mutex<HashMap<String, u32>>,
    cached_pipelines: Mutex<HashMap<usize, (Arc<str>, u32)>>,
    profiling: bool,
    pub supports_subgroups: bool,
    pub subgroup_size_range: Option<(u32, u32)>,
    pub subgroup_id_supported: bool,
}

impl Runtime {
    pub fn initialize_from_source(
        &self,
        id: u32,
        source: &JsValue,
        offset: u64,
        size: usize,
        cancellation: &JsValue,
    ) -> Result<(), WebGpuError> {
        initialize_buffer_from_source(self.id, id, source, offset as f64, size, cancellation).map_err(Into::into)
    }

    pub fn record_host_allocation(
        &self,
        size: usize,
    ) {
        if self.profiling {
            record_host_allocation(self.id, size);
        }
    }

    pub fn new() -> Result<Arc<Self>, WebGpuError> {
        let info: DeviceInfo =
            serde_json::from_str(&create_context()?).map_err(|error| WebGpuError::Runtime(error.to_string()))?;
        Ok(Arc::new(Self {
            id: info.id,
            name: info.name,
            max_buffer_size: info.max_buffer_size.min(info.max_storage_buffer_binding_size),
            next_buffer: AtomicU32::new(1),
            pipelines: Mutex::new(HashMap::new()),
            cached_pipelines: Mutex::new(HashMap::new()),
            profiling: info.profiling,
            supports_subgroups: info.supports_subgroups,
            subgroup_size_range: (info.subgroup_min_size > 0 && info.subgroup_max_size > 0)
                .then_some((info.subgroup_min_size, info.subgroup_max_size)),
            subgroup_id_supported: info.subgroup_id_supported,
        }))
    }

    pub fn next_buffer_id(&self) -> u32 {
        self.next_buffer.fetch_add(1, Ordering::Relaxed)
    }

    pub fn profile_start(&self) -> Option<f64> {
        self.profiling.then(profile_now)
    }

    pub fn profile_end(
        &self,
        stage: &str,
        started: Option<f64>,
    ) {
        if let Some(started) = started {
            profile_elapsed(self.id, stage, started);
        }
    }

    pub fn create_buffer(
        &self,
        id: u32,
        size: usize,
    ) -> Result<(), WebGpuError> {
        create_buffer(self.id, id, size).map_err(Into::into)
    }

    pub fn destroy_buffer(
        &self,
        id: u32,
    ) {
        destroy_buffer(self.id, id);
    }

    pub fn upload(
        &self,
        id: u32,
        data: &[u8],
    ) -> Result<(), WebGpuError> {
        upload_buffer(self.id, id, data).map_err(Into::into)
    }

    pub fn read(
        &self,
        id: u32,
        data: &mut [u8],
    ) -> Result<(), WebGpuError> {
        read_buffer(self.id, id, data).map_err(Into::into)
    }

    pub fn pipeline(
        &self,
        source: &str,
    ) -> Result<u32, WebGpuError> {
        let mut pipelines = self.pipelines.lock();
        if let Some(id) = pipelines.get(source) {
            return Ok(*id);
        }
        let id = pipelines.len() as u32 + 1;
        register_pipeline(self.id, id, source)?;
        pipelines.insert(source.to_owned(), id);
        Ok(id)
    }

    pub fn pipeline_cached(
        &self,
        source: &Arc<str>,
    ) -> Result<u32, WebGpuError> {
        let key = Arc::as_ptr(source).cast::<u8>() as usize;
        let mut cache = self.cached_pipelines.lock();
        if let Some((_, id)) = cache.get(&key) {
            return Ok(*id);
        }
        let id = self.pipeline(source)?;
        // Retaining the source prevents allocator address reuse from ever
        // associating a different shader with this pointer's cached pipeline.
        cache.insert(key, (source.clone(), id));
        Ok(id)
    }

    pub fn submit(
        &self,
        commands: &[u32],
        written_buffers: &[u32],
    ) -> Result<u32, WebGpuError> {
        submit_commands(self.id, commands, written_buffers).map_err(Into::into)
    }

    pub fn wait(
        &self,
        submission: u32,
    ) -> Result<f64, WebGpuError> {
        wait_commands(self.id, submission).map_err(Into::into)
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        destroy_context(self.id);
    }
}
