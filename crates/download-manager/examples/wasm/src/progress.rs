use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen(getter_with_clone)]
pub struct JsFileDownloadState {
    pub task_id: String,
    pub phase: String,
    pub downloaded_bytes: f64,
    pub total_bytes: f64,
    pub message: Option<String>,
}
