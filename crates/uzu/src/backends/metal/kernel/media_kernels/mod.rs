use std::mem;

use metal::{
    Buffer as MTLBuffer, CommandBuffer as MTLCommandBuffer,
    ComputeCommandEncoder, ComputePipelineState, MTLSize,
    foreign_types::{ForeignType, ForeignTypeRef},
};
use mpsgraph::CommandBuffer as MPSCommandBuffer;

use crate::backends::metal::{
    MTLContext,
    error::MTLError,
    forward_pass::{
        ForwardPassState,
        encodable_with_state::{EncodableWithState, EncodingParameters},
    },
    image::Image,
    metal_extensions::ComputeEncoderDispatch,
};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ImageParameters {
    pub input_dimensions: [u32; 2],
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub padding_value_rgb: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PatchParameters {
    pub padded_dimensions: [u32; 2],
    pub patch_size: u32,
    pub num_channels: u32,
    pub temporal_slices: u32,
}

#[derive(Debug)]
pub struct ScalePadNormalizeImage {
    pipeline_state: ComputePipelineState,
}

impl ScalePadNormalizeImage {
    pub fn new(context: &MTLContext) -> Result<Self, MTLError> {
        let function_name = "scalePadNormalizeImage";
        let (pipeline_state, _argument_names) = context
            .compute_pipeline_state_with_reflection(&function_name, None)?;
        Ok(Self {
            pipeline_state,
        })
    }

    pub fn encode_internal(
        &self,
        input_image: &Image,
        output_image: &Image,
        image_params_ptr: *const std::ffi::c_void,
        command_buffer: &MTLCommandBuffer,
    ) {
        let compute_encoder = unsafe {
            ComputeCommandEncoder::from_ptr(
                command_buffer.new_compute_command_encoder().as_ptr(),
            )
        };
        compute_encoder.set_label("ScalePadNormalizeImageEncoder");

        compute_encoder.set_compute_pipeline_state(&self.pipeline_state);
        compute_encoder.set_texture(0, Some(input_image.texture_ref()));
        compute_encoder.set_texture(1, Some(output_image.texture_ref()));
        compute_encoder.set_bytes(
            0,
            mem::size_of::<ImageParameters>() as u64,
            image_params_ptr,
        );

        let grid_size = MTLSize {
            width: output_image.width() as u64,
            height: output_image.height() as u64,
            depth: 1,
        };

        compute_encoder.dispatch_2d(&self.pipeline_state, grid_size, None);
        compute_encoder.end_encoding();
    }
}

impl EncodableWithState for ScalePadNormalizeImage {
    fn encode(
        &self,
        _state: &mut ForwardPassState,
        command_buffer: &MPSCommandBuffer,
        _parameters: &EncodingParameters,
    ) {
        let _ = command_buffer;
    }

    fn supports_shared_encoder(&self) -> bool {
        false // Image processing uses its own encoding path
    }
}

#[derive(Debug)]
pub struct ExtractImagePatches {
    pipeline_state: ComputePipelineState,
}

impl ExtractImagePatches {
    pub fn new(context: &MTLContext) -> Result<Self, MTLError> {
        let function_name = "extractImagePatches";
        let (pipeline_state, _argument_names) = context
            .compute_pipeline_state_with_reflection(&function_name, None)?;
        Ok(Self {
            pipeline_state,
        })
    }

    pub fn encode_internal(
        &self,
        padded_normalized_image: &Image,
        output_buffer_mtl: &MTLBuffer,
        patch_params_ptr: *const std::ffi::c_void,
        command_buffer: &MTLCommandBuffer,
    ) {
        let compute_encoder =
            command_buffer.new_compute_command_encoder().to_owned();
        compute_encoder.set_label("ExtractImagePatchesEncoder");

        compute_encoder.set_compute_pipeline_state(&self.pipeline_state);
        compute_encoder
            .set_texture(0, Some(padded_normalized_image.texture_ref()));
        compute_encoder.set_buffer(0, Some(output_buffer_mtl), 0);
        compute_encoder.set_bytes(
            1,
            mem::size_of::<PatchParameters>() as u64,
            patch_params_ptr,
        );

        let grid_size = MTLSize {
            width: padded_normalized_image.width() as u64,
            height: padded_normalized_image.height() as u64,
            depth: 1,
        };

        compute_encoder.dispatch_2d(&self.pipeline_state, grid_size, None);
        compute_encoder.end_encoding();
    }
}
