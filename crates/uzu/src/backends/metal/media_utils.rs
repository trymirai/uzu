use std::{cell::RefCell, mem, rc::Rc};

use crate::backends::metal::{
    MTLCommandBuffer, MTLDevice, MTLResourceOptions, ProtocolObject, Retained,
};
use metal::MTLDeviceExt;

use crate::{
    DataType,
    backends::metal::{
        MTLContext, MetalArray,
        error::MTLError,
        image::{Image, PixelFormat, TextureUsage},
        kernel::media_kernels::{
            ExtractImagePatches, ImageParameters, PatchParameters,
            ScalePadNormalizeImage,
        },
    },
};

/// Calculate scaled and padded dimensions for preprocessing input images.
/// This function handles resizing constraints and ensures dimensions are
/// divisible by patch_size.
fn calculate_preprocessing_dimensions(
    input_width: u32,
    input_height: u32,
    min_pixels: u32,
    max_pixels: u32,
    patch_size: u32,
) -> (u32, u32, f32) {
    if input_width == 0 || input_height == 0 {
        return (patch_size, patch_size, 1.0);
    }

    let original_pixels = (input_width * input_height) as f32;
    let min_pixels_f = min_pixels as f32;
    let max_pixels_f = max_pixels as f32;

    let mut scale_factor = 1.0;

    if original_pixels < min_pixels_f {
        scale_factor = (min_pixels_f / original_pixels).sqrt();
    } else if original_pixels > max_pixels_f {
        scale_factor = (max_pixels_f / original_pixels).sqrt();
    }

    let scaled_width = (input_width as f32 * scale_factor).ceil() as u32;
    let scaled_height = (input_height as f32 * scale_factor).ceil() as u32;

    let padded_width =
        ((scaled_width + patch_size - 1) / patch_size) * patch_size;
    let padded_height =
        ((scaled_height + patch_size - 1) / patch_size) * patch_size;

    (padded_width, padded_height, scale_factor)
}

#[derive(Debug, Clone)]
pub struct ImagePreprocessingParams {
    pub min_pixels: u32,
    pub max_pixels: u32,
    pub patch_size: u32,
    pub num_channels: u32,
    pub temporal_slices: u32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub padding_value_rgb: f32,
}

impl Default for ImagePreprocessingParams {
    fn default() -> Self {
        Self {
            min_pixels: 224 * 224,
            max_pixels: 1024 * 1024,
            patch_size: 14,
            num_channels: 3,
            temporal_slices: 2,
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.26862954, 0.26130258, 0.27577711],
            padding_value_rgb: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImagePreprocessingRequirements {
    pub padded_width: u32,
    pub padded_height: u32,
    pub output_buffer_element_count: u32,
    pub intermediate_texture_pixel_format: PixelFormat,
    pub intermediate_texture_usage: TextureUsage,
    pub output_buffer_data_type: DataType,
}

pub fn calculate_image_preprocessing_requirements(
    input_width: u32,
    input_height: u32,
    params: &ImagePreprocessingParams,
) -> ImagePreprocessingRequirements {
    let (padded_width, padded_height, _scale_factor) =
        calculate_preprocessing_dimensions(
            input_width,
            input_height,
            params.min_pixels,
            params.max_pixels,
            params.patch_size,
        );

    let num_patches_x = padded_width / params.patch_size;
    let num_patches_y = padded_height / params.patch_size;
    let total_spatial_patches = num_patches_x * num_patches_y;
    let output_buffer_element_count = total_spatial_patches
        * params.temporal_slices
        * params.num_channels
        * params.patch_size
        * params.patch_size;

    let intermediate_texture_usage = TextureUsage {
        shader_read: true,
        shader_write: true,
        render_target: false,
        pixel_format_view: false,
    };

    ImagePreprocessingRequirements {
        padded_width,
        padded_height,
        output_buffer_element_count,
        intermediate_texture_pixel_format: PixelFormat::RGBA32Float,
        intermediate_texture_usage,
        output_buffer_data_type: DataType::F32,
    }
}

pub struct MetalImagePreprocessor {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    scale_pad_normalize_kernel: ScalePadNormalizeImage,
    extract_patches_kernel: ExtractImagePatches,
}

impl MetalImagePreprocessor {
    pub fn new(context: &MTLContext) -> Result<Self, MTLError> {
        let scale_pad_normalize_kernel = ScalePadNormalizeImage::new(context)?;
        let extract_patches_kernel = ExtractImagePatches::new(context)?;
        Ok(Self {
            device: context.device.clone(),
            scale_pad_normalize_kernel,
            extract_patches_kernel,
        })
    }

    pub unsafe fn encode_preprocessing(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        input_image: &Image,
        params: &ImagePreprocessingParams,
        requirements: &ImagePreprocessingRequirements,
        intermediate_texture: &Image,
        output_patch_buffer_cell: Rc<RefCell<MetalArray>>,
    ) -> Result<(), MTLError> {
        let padded_width = requirements.padded_width;
        let padded_height = requirements.padded_height;

        let input_dimensions_data = [input_image.width(), input_image.height()];
        let mut input_dimensions_tensor = data_to_mtl_tensor(
            &self.device,
            &input_dimensions_data,
            vec![2],
            DataType::F32,
        )?;
        let mut image_mean_tensor = data_to_mtl_tensor(
            &self.device,
            &params.image_mean,
            vec![3],
            DataType::F32,
        )?;
        let mut image_std_tensor = data_to_mtl_tensor(
            &self.device,
            &params.image_std,
            vec![3],
            DataType::F32,
        )?;
        let mut padding_value_rgb_tensor = data_to_mtl_tensor(
            &self.device,
            &params.padding_value_rgb,
            vec![1],
            DataType::F32,
        )?;

        let _spn_tensors: Vec<&mut MetalArray> = vec![
            &mut input_dimensions_tensor,
            &mut image_mean_tensor,
            &mut image_std_tensor,
            &mut padding_value_rgb_tensor,
        ];
        let _spn_images = [input_image, intermediate_texture];

        let image_params = ImageParameters {
            input_dimensions: [input_image.width(), input_image.height()],
            image_mean: params.image_mean,
            image_std: params.image_std,
            padding_value_rgb: params.padding_value_rgb,
        };
        self.scale_pad_normalize_kernel.encode_internal(
            input_image,
            intermediate_texture,
            &image_params as *const _ as *const std::ffi::c_void,
            command_buffer,
        );

        let mut final_output_buffer_array_ref =
            output_patch_buffer_cell.borrow_mut();

        let padded_dimensions_data = [padded_width, padded_height];
        let mut padded_dimensions_tensor = data_to_mtl_tensor(
            &self.device,
            &padded_dimensions_data,
            vec![2],
            DataType::F32,
        )?;
        let mut patch_size_tensor = data_to_mtl_tensor(
            &self.device,
            &params.patch_size,
            vec![1],
            DataType::F32,
        )?;
        let mut num_channels_tensor = data_to_mtl_tensor(
            &self.device,
            &params.num_channels,
            vec![1],
            DataType::F32,
        )?;
        let mut temporal_slices_tensor = data_to_mtl_tensor(
            &self.device,
            &params.temporal_slices,
            vec![1],
            DataType::F32,
        )?;

        let _eip_tensors: Vec<&mut MetalArray> = vec![
            &mut *final_output_buffer_array_ref,
            &mut padded_dimensions_tensor,
            &mut patch_size_tensor,
            &mut num_channels_tensor,
            &mut temporal_slices_tensor,
        ];
        let _eip_images = [intermediate_texture];

        let patch_params = PatchParameters {
            padded_dimensions: [
                requirements.padded_width,
                requirements.padded_height,
            ],
            patch_size: params.patch_size,
            num_channels: params.num_channels,
            temporal_slices: params.temporal_slices,
        };
        self.extract_patches_kernel.encode_internal(
            intermediate_texture,
            unsafe { final_output_buffer_array_ref.mtl_buffer() },
            &patch_params as *const _ as *const std::ffi::c_void,
            command_buffer,
        );

        Ok(())
    }
}

fn data_to_mtl_tensor<T: Copy>(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    data: &T,
    shape: Vec<usize>,
    data_type: DataType,
) -> Result<MetalArray, MTLError> {
    let size = mem::size_of::<T>();
    let bytes = unsafe { std::slice::from_raw_parts(data as *const T as *const u8, size) };
    let buffer = device.new_buffer_with_data(
        bytes,
        MTLResourceOptions::STORAGE_MODE_SHARED,
    ).expect("Failed to create buffer");
    unsafe { Ok(MetalArray::new(buffer, &shape, data_type)) }
}
