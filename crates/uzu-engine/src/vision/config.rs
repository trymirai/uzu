use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionModelConfig {
    #[serde(default)]
    pub vision_config: VisionConfig,
    #[serde(default)]
    pub canvas_config: CanvasConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_window_kernel_size")]
    pub window_kernel_size: [usize; 2],
}

fn default_hidden_size() -> usize { 1152 }
fn default_intermediate_size() -> usize { 4304 }
fn default_num_hidden_layers() -> usize { 27 }
fn default_num_attention_heads() -> usize { 16 }
fn default_num_channels() -> usize { 3 }
fn default_image_size() -> usize { 980 }
fn default_patch_size() -> usize { 14 }
fn default_hidden_act() -> String { "gelu_pytorch_tanh".to_string() }
fn default_layer_norm_eps() -> f32 { 1e-6 }
fn default_window_kernel_size() -> [usize; 2] { [2, 2] }

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_channels: default_num_channels(),
            image_size: default_image_size(),
            patch_size: default_patch_size(),
            hidden_act: default_hidden_act(),
            layer_norm_eps: default_layer_norm_eps(),
            window_kernel_size: default_window_kernel_size(),
        }
    }
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanvasConfig {
    #[serde(default = "default_insert_layer_id")]
    pub insert_layer_id: usize,
    #[serde(default = "default_downsample_mode")]
    pub downsample_mode: String,
    #[serde(default = "default_mrope_mode")]
    pub mrope_mode: String,
    #[serde(default = "default_window_kernel_size")]
    pub merge_kernel_size: [usize; 2],
    #[serde(default = "default_merger_times")]
    pub merger_times: usize,
    #[serde(default = "default_image_start_id")]
    pub image_start_id: u32,
    #[serde(default = "default_image_end_id")]
    pub image_end_id: u32,
    #[serde(default = "default_slice_start_id")]
    pub slice_start_id: u32,
    #[serde(default = "default_slice_end_id")]
    pub slice_end_id: u32,
    #[serde(default = "default_image_id_start")]
    pub image_id_start: u32,
    #[serde(default = "default_image_id_end")]
    pub image_id_end: u32,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    #[serde(default = "default_newline_id")]
    pub newline_id: u32,
}

fn default_insert_layer_id() -> usize { 6 }
fn default_downsample_mode() -> String { "16x".to_string() }
fn default_mrope_mode() -> String { "canvas".to_string() }
fn default_merger_times() -> usize { 1 }
fn default_image_start_id() -> u32 { 248078 }
fn default_image_end_id() -> u32 { 248079 }
fn default_slice_start_id() -> u32 { 248088 }
fn default_slice_end_id() -> u32 { 248089 }
fn default_image_id_start() -> u32 { 248090 }
fn default_image_id_end() -> u32 { 248091 }
fn default_image_token_id() -> u32 { 248056 }
fn default_newline_id() -> u32 { 198 } // Note: 198 is standard \n token in Qwen2/MiniCPM tokenizer

impl Default for CanvasConfig {
    fn default() -> Self {
        Self {
            insert_layer_id: default_insert_layer_id(),
            downsample_mode: default_downsample_mode(),
            mrope_mode: default_mrope_mode(),
            merge_kernel_size: default_window_kernel_size(),
            merger_times: default_merger_times(),
            image_start_id: default_image_start_id(),
            image_end_id: default_image_end_id(),
            slice_start_id: default_slice_start_id(),
            slice_end_id: default_slice_end_id(),
            image_id_start: default_image_id_start(),
            image_id_end: default_image_id_end(),
            image_token_id: default_image_token_id(),
            newline_id: default_newline_id(),
        }
    }
}

impl CanvasConfig {
    pub fn merge_factor(&self) -> usize {
        if self.downsample_mode == "4x" { 2 } else { 4 }
    }

    /// Dynamically resolve special token IDs from tokenizer if present
    pub fn resolve_from_tokenizer(&mut self, tokenizer: &tokenizers::Tokenizer) {
        if let Some(id) = tokenizer.token_to_id("<|image_pad|>") {
            self.image_token_id = id;
        }
        if let Some(id) = tokenizer.token_to_id("<image>") {
            self.image_start_id = id;
        }
        if let Some(id) = tokenizer.token_to_id("</image>") {
            self.image_end_id = id;
        }
        if let Some(id) = tokenizer.token_to_id("<slice>") {
            self.slice_start_id = id;
        }
        if let Some(id) = tokenizer.token_to_id("</slice>") {
            self.slice_end_id = id;
        }
        if let Some(id) = tokenizer.token_to_id("<image_id>") {
            self.image_id_start = id;
        }
        if let Some(id) = tokenizer.token_to_id("</image_id>") {
            self.image_id_end = id;
        }
        if let Ok(encoding) = tokenizer.encode("\n", false)
            && let Some(&id) = encoding.get_ids().first() {
                self.newline_id = id;
            }
    }
}
