#![allow(dead_code)]
use std::{
    cell::RefCell, fs::File, io::BufReader, path::Path, rc::Rc, time::Instant,
};

use mpsgraph::CommandBuffer as MPSCommandBuffer;
use serde::{Deserialize, Serialize};

use crate::{
    backends::metal::{
        DecoderExecutables, ForwardPassState, KVCache, MTLContext, ModelShape,
        compilation_parameters::CompilationConfig,
        forward_pass::{
            ForwardPassBuffers, SharedBuffers,
            encodable_with_state::{EncodableWithState, EncodingParameters},
        },
        utils::{CaptureOptions, StdoutCapture},
    },
    config::{ModelConfig, decoder::DecoderConfig},
    parameters::ParameterLoader,
};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderTestResult {
    pub placement_log: String,
    pub iterations: u32,
    pub time_per_token: f32,
    pub tokens_per_second: f32,
    pub success: bool,
    pub error: Option<String>,
}

struct DecoderTestContext {
    mtl_context: Rc<MTLContext>,
    mtl_command_queue: metal::CommandQueue,
    state: ForwardPassState,
    executables: DecoderExecutables,
    suffix_length: usize,
    _scratch: ForwardPassBuffers,
}

impl DecoderTestContext {
    fn new(model_dir: &str) -> Result<(Self, Rc<DecoderConfig>), String> {
        let model_path = Path::new(model_dir);
        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(format!(
                "Decoder config file not found at {:?}",
                config_path
            ));
        }
        let config_file = File::open(&config_path)
            .map_err(|e| format!("Failed to open config: {}", e))?;
        let model_config: ModelConfig =
            serde_json::from_reader(BufReader::new(config_file))
                .map_err(|e| format!("Failed to parse config: {}", e))?;
        let decoder_config = Rc::new(model_config.model_config.clone());
        let mtl_device = metal::Device::system_default()
            .ok_or("No Metal device available".to_string())?;
        let mtl_command_queue = mtl_device.new_command_queue();

        let compilation_config = Rc::new(CompilationConfig::default());

        // -----------------------------------------------------------------
        // Determine the heap size we need for the worst-case forward pass
        // *and* all parameters that will be loaded from the safetensors file.
        // -----------------------------------------------------------------
        let max_prefix_length = 8192; // Example value, can be passed dynamically
        let max_suffix_length = 1; // Example value, can be passed dynamically

        let model_shape = ModelShape::from_decoder_config(&decoder_config);

        // Read safetensors header to know the total size of all tensors
        // so we can make the heap large enough to store them.
        let weights_path = model_path.join("model.safetensors");
        if !weights_path.exists() {
            return Err(format!(
                "Decoder weights file not found at {:?}",
                weights_path
            ));
        }

        let mtl_context = Rc::new(
            MTLContext::new(mtl_device.clone(), mtl_command_queue.clone())
                .map_err(|e| format!("Failed to create MetalContext: {}", e))?,
        );

        let weights_file = File::open(&weights_path)
            .map_err(|e| format!("Failed to open weights: {}", e))?;
        let loader = ParameterLoader::new(&weights_file, &mtl_context)
            .map_err(|e| format!("Failed to create ParameterLoader: {}", e))?;
        let root_loader_view = loader.tree();

        let kv_cache = Rc::new(RefCell::new(KVCache::new(
            &mtl_context,
            &model_shape,
            max_prefix_length,
            max_suffix_length,
        )));
        let shared_buffers = Rc::new(RefCell::new(SharedBuffers::new(
            &mtl_context,
            &decoder_config,
            &model_shape,
        )));
        shared_buffers.borrow_mut().update_data(&root_loader_view);
        let token_ids: Vec<u64> =
            (0..max_suffix_length).map(|i| (i % 1000) as u64).collect();
        let token_positions: Vec<usize> = (0..max_suffix_length).collect();

        // Scratch buffers sized for the maximum prefix/suffix lengths in this test
        let scratch_buffers = ForwardPassBuffers::new(
            &mtl_context,
            &model_shape,
            max_prefix_length,
            max_suffix_length,
        );

        let state = ForwardPassState::new(
            mtl_context.clone(),
            &model_shape,
            &scratch_buffers,
            kv_cache.clone(),
            shared_buffers.clone(),
            &token_ids,
            &token_positions,
            false,
            None,
        );

        let executables = DecoderExecutables::new(
            mtl_context.clone(),
            decoder_config.clone(),
            &root_loader_view,
            compilation_config.clone(),
        );
        Ok((
            Self {
                mtl_context,
                mtl_command_queue,
                state,
                executables,
                suffix_length: max_suffix_length,
                _scratch: scratch_buffers,
            },
            decoder_config,
        ))
    }

    fn run_warmup_with_placement_analysis(&mut self) -> String {
        let mut capture = StdoutCapture::new(CaptureOptions::STDERR);
        capture.start();
        let command_buffer =
            MPSCommandBuffer::from_command_queue(&self.mtl_command_queue);
        self.executables.encode(
            &mut self.state,
            &command_buffer,
            &EncodingParameters::new(true, true, false),
        );
        command_buffer.commit();
        command_buffer.command_buffer().wait_until_completed();
        capture.stop()
    }

    fn run_timing_loop(
        &mut self,
        iterations: u32,
    ) -> Result<std::time::Duration, String> {
        // Warmup run (not measured)
        let command_buffer =
            MPSCommandBuffer::from_command_queue(&self.mtl_command_queue);
        self.executables.encode(
            &mut self.state,
            &command_buffer,
            &EncodingParameters::new(true, true, false),
        );
        command_buffer.commit();
        command_buffer.command_buffer().wait_until_completed();
        // Measured runs
        let start = Instant::now();
        for _ in 0..iterations {
            let command_buffer =
                MPSCommandBuffer::from_command_queue(&self.mtl_command_queue);
            self.executables.encode(
                &mut self.state,
                &command_buffer,
                &EncodingParameters::new(false, true, false),
            );
            command_buffer.commit();
            command_buffer.command_buffer().wait_until_completed();
        }
        Ok(start.elapsed())
    }
}

pub fn run_decoder_with_results(model_dir: &str) -> DecoderTestResult {
    let mut result = DecoderTestResult {
        placement_log: String::new(),
        iterations: 0,
        time_per_token: 0.0,
        tokens_per_second: 0.0,
        success: false,
        error: None,
    };
    // Setup context for both placement analysis and timing
    let (mut ctx, _decoder_config) = match DecoderTestContext::new(model_dir) {
        Ok(pair) => pair,
        Err(e) => {
            result.error = Some(e);
            return result;
        },
    };
    // Warmup run with placement analysis
    let placement_log = ctx.run_warmup_with_placement_analysis();
    result.placement_log = placement_log;
    // After warmup, turn off placement analysis for timing loop
    // (Assume placement analysis only affects compilation, not execution)
    let iterations = 100;
    match ctx.run_timing_loop(iterations) {
        Ok(duration) => {
            result.iterations = iterations;
            result.time_per_token = duration.as_secs_f32() / iterations as f32;
            result.tokens_per_second = 1.0 / result.time_per_token;
            result.success = true;
        },
        Err(e) => {
            result.error = Some(e);
        },
    }
    result
}
