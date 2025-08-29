#![allow(dead_code)]
use std::{cell::RefCell, fs::File, path::Path, rc::Rc, time::Instant};

use mpsgraph::CommandBuffer as MPSCommandBuffer;
use serde::{Deserialize, Serialize};

use crate::{
    backends::{
        KVCache, MetalBackend,
        metal::{
            DecoderExecutables, ForwardPassState, MTLContext, ModelShape,
            compilation_parameters::CompilationConfig,
            forward_pass::{
                ForwardPassBuffers, SharedBuffers,
                encodable_with_state::{
                    EncodableWithState, EncodingParameters,
                },
            },
            utils::{CaptureOptions, StdoutCapture},
        },
    },
    config::decoder::DecoderConfig,
    parameters::ParameterLoader,
    utils::{load_decoder_config, open_weights_file},
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

struct DecoderTestContext<'c> {
    context: &'c MTLContext,
    state: ForwardPassState<'c>,
    executables: DecoderExecutables,
    suffix_length: usize,
    _scratch: ForwardPassBuffers,
}

// ----------------------------------------------------------------
// Determine the heap size we need for the worst-case forward pass
// *and* all parameters that will be loaded from the safetensors file.
// -----------------------------------------------------------------
const MAX_PREFIX_LENGTH: usize = 8192; // Example value, can be passed dynamically
const MAX_SUFFIX_LENGTH: usize = 1; // Example value, can be passed dynamically

impl<'c> DecoderTestContext<'c> {
    fn new(
        mtl_context: &'c MTLContext,
        decoder_config: &DecoderConfig,
        weights_file: &File,
        kv_cache: &'c mut KVCache<MetalBackend>,
    ) -> Result<Self, String> {
        let compilation_config = Rc::new(CompilationConfig::default());
        let model_shape = ModelShape::from_decoder_config(&decoder_config);

        // Read safetensors header to know the total size of all tensors
        // so we can make the heap large enough to store them.
        let loader = ParameterLoader::new(&weights_file, mtl_context)
            .map_err(|e| format!("Failed to create ParameterLoader: {}", e))?;
        let root_loader_view = loader.tree();

        let shared_buffers = Rc::new(RefCell::new(SharedBuffers::new(
            &mtl_context,
            &decoder_config,
            &model_shape,
        )));
        shared_buffers.borrow_mut().update_data(&root_loader_view);
        let token_ids: Vec<u64> =
            (0..MAX_SUFFIX_LENGTH).map(|i| (i % 1000) as u64).collect();
        let token_positions: Vec<usize> = (0..MAX_SUFFIX_LENGTH).collect();

        // Scratch buffers sized for the maximum prefix/suffix lengths in this test
        let scratch_buffers = ForwardPassBuffers::new(
            &mtl_context,
            &model_shape,
            MAX_PREFIX_LENGTH,
            MAX_SUFFIX_LENGTH,
        );

        let state = ForwardPassState::new(
            mtl_context,
            &model_shape,
            &scratch_buffers,
            kv_cache,
            shared_buffers.clone(),
            &token_ids,
            &token_positions,
            false,
            None,
        );

        let executables = DecoderExecutables::new(
            mtl_context,
            &decoder_config,
            &root_loader_view,
            compilation_config.clone(),
        );
        Ok(Self {
            context: mtl_context,
            state,
            executables,
            suffix_length: MAX_SUFFIX_LENGTH,
            _scratch: scratch_buffers,
        })
    }

    fn run_warmup_with_placement_analysis(&mut self) -> String {
        let mut capture = StdoutCapture::new(CaptureOptions::STDERR);
        capture.start();
        let command_buffer =
            MPSCommandBuffer::from_command_queue(&self.context.command_queue);
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
            MPSCommandBuffer::from_command_queue(&self.context.command_queue);
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
            let command_buffer = MPSCommandBuffer::from_command_queue(
                &self.context.command_queue,
            );
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

fn load_model(model_path: &Path) -> Result<(DecoderConfig, File), String> {
    Ok((load_decoder_config(model_path)?, open_weights_file(model_path)?))
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

    let mtl_device =
        metal::Device::system_default().expect("No Metal device available");
    let mtl_command_queue = mtl_device.new_command_queue();

    let mtl_context =
        MTLContext::new(mtl_device.clone(), mtl_command_queue.clone())
            .expect("Metal context creation failed");

    let model_path = Path::new(model_dir);

    let (decoder_config, weigths_file) = match load_model(model_path) {
        Ok(pair) => pair,
        Err(e) => {
            result.error = Some(e);
            return result;
        },
    };

    let mut kv_cache = Box::new(KVCache::new(
        &mtl_context,
        &ModelShape::from_decoder_config(&decoder_config),
        MAX_PREFIX_LENGTH,
        MAX_SUFFIX_LENGTH,
    ));

    // Setup context for both placement analysis and timing
    let mut ctx = match DecoderTestContext::new(
        &mtl_context,
        &decoder_config,
        &weigths_file,
        &mut kv_cache,
    ) {
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
