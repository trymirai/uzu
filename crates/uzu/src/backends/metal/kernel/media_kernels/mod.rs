use std::{mem, ptr::NonNull};

use objc2::rc::Retained;

use crate::backends::metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoderExt,
    MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePipelineState, MTLContext, MTLSize,
    ProtocolObject,
    error::MTLError,
    forward_pass::{EncodableBlock, EncodingParameters, ForwardPassState},
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
    pipeline_state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) {
        let compute_encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        compute_encoder.set_label(Some("ScalePadNormalizeImageEncoder"));

        compute_encoder.set_compute_pipeline_state(&self.pipeline_state);
        compute_encoder.set_texture(Some(input_image.texture_ref()), 0);
        compute_encoder.set_texture(Some(output_image.texture_ref()), 1);
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new_unchecked(image_params_ptr as *mut _),
                mem::size_of::<ImageParameters>(),
                0,
            );
        }

        let grid_size = MTLSize {
            width: output_image.width() as usize,
            height: output_image.height() as usize,
            depth: 1,
        };

        compute_encoder.dispatch_2d(&self.pipeline_state, grid_size, None);
        compute_encoder.end_encoding();
    }
}

impl EncodableBlock for ScalePadNormalizeImage {
    fn encode(
        &self,
        _state: &mut ForwardPassState,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        _parameters: &EncodingParameters,
    ) {
        let _ = command_buffer;
    }

    fn supports_shared_encoder(&self) -> bool {
        false // Image processing uses its own encoding path
    }

    fn encode_with_shared_encoder(
        &self,
        _state: &mut ForwardPassState,
        _encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        _parameters: &EncodingParameters,
    ) {
        unreachable!(
            "ScalePadNormalizeImage does not support shared compute encoder"
        );
    }
}

#[derive(Debug)]
pub struct ExtractImagePatches {
    pipeline_state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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
        output_buffer_mtl: &ProtocolObject<dyn MTLBuffer>,
        patch_params_ptr: *const std::ffi::c_void,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) {
        let compute_encoder = command_buffer
            .new_compute_command_encoder()
            .expect("Failed to create compute command encoder");
        compute_encoder.set_label(Some("ExtractImagePatchesEncoder"));

        compute_encoder.set_compute_pipeline_state(&self.pipeline_state);
        compute_encoder
            .set_texture(Some(padded_normalized_image.texture_ref()), 0);
        compute_encoder.set_buffer(Some(output_buffer_mtl), 0, 0);
        unsafe {
            compute_encoder.set_bytes(
                NonNull::new_unchecked(patch_params_ptr as *mut _),
                mem::size_of::<PatchParameters>(),
                1,
            );
        }

        let grid_size = MTLSize {
            width: padded_normalized_image.width() as usize,
            height: padded_normalized_image.height() as usize,
            depth: 1,
        };

        compute_encoder.dispatch_2d(&self.pipeline_state, grid_size, None);
        compute_encoder.end_encoding();
    }
}
