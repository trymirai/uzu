use std::io::Cursor;

use image::{
    DynamicImage, GrayAlphaImage, GrayImage, ImageDecoder, ImageError, ImageReader, RgbImage, RgbaImage, imageops,
    imageops::FilterType,
};
use zune_heic::{
    HeicErrors, HeifDecoder,
    zune_core::{bytestream::ZCursor, colorspace::ColorSpace},
};

const MAX_ASPECT_RATIO: f64 = 200.0;

#[derive(Debug, thiserror::Error)]
pub enum VisionPreprocessingError {
    #[error("Aspect ratio too large: {width}x{height}, max ratio: {max_ratio}")]
    AspectRatio {
        width: u32,
        height: u32,
        max_ratio: f64,
    },
    #[error(transparent)]
    Image(#[from] ImageError),
    #[error(transparent)]
    Heic(#[from] HeicErrors),
    #[error("Unsupported HEIC pixel format: {0:?}")]
    HeicPixelFormat(Option<ColorSpace>),
}

pub struct VisionPreprocessingConfig {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    /// Minimum pixel count (width * height), HF `size.shortest_edge`.
    pub shortest_edge: usize,
    /// Maximum pixel count (width * height), HF `size.longest_edge`.
    pub longest_edge: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

pub struct VisionPreprocessingOutput {
    /// `[grid_temporal * grid_height * grid_width, 3 * temporal_patch_size * patch_size * patch_size]`
    pub pixel_values: Vec<f32>,
    /// Patch grid `[temporal, height, width]` before merging
    pub image_grid_thw: [usize; 3],
}

pub fn preprocess(
    image_bytes: &[u8],
    config: &VisionPreprocessingConfig,
) -> Result<VisionPreprocessingOutput, VisionPreprocessingError> {
    let factor_size = config.merge_size * config.patch_size;
    let image = get_rgb_image(image_bytes)?;
    let (new_width, new_height) = get_new_size(&image, factor_size, config.shortest_edge, config.longest_edge)?;
    let image = imageops::resize(&image, new_width as u32, new_height as u32, FilterType::CatmullRom);
    let normalized = normalize(&image, config.image_mean, config.image_std);
    let pixel_values =
        patchify(&normalized, new_width, new_height, config.patch_size, config.merge_size, config.temporal_patch_size);
    Ok(VisionPreprocessingOutput {
        pixel_values,
        image_grid_thw: [1, new_height / config.patch_size, new_width / config.patch_size],
    })
}

fn get_rgb_image(image_bytes: &[u8]) -> Result<RgbImage, VisionPreprocessingError> {
    let cursor = Cursor::new(image_bytes);
    let reader = ImageReader::new(cursor).with_guessed_format().map_err(ImageError::from)?;
    if reader.format().is_none() && image_bytes.get(4..8) == Some(b"ftyp".as_slice()) {
        return get_heic_rgb_image(image_bytes);
    }

    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image.into_rgb8())
}

fn get_heic_rgb_image(image_bytes: &[u8]) -> Result<RgbImage, VisionPreprocessingError> {
    let mut decoder = HeifDecoder::new(ZCursor::new(image_bytes));
    let pixels = decoder.decode()?;
    let colorspace = decoder.colorspace();
    let (Some(width), Some(height)) = (decoder.width(), decoder.height()) else {
        return Err(VisionPreprocessingError::HeicPixelFormat(colorspace));
    };
    let (width, height) = (width as u32, height as u32);
    let image = match colorspace {
        Some(ColorSpace::RGB) => RgbImage::from_raw(width, height, pixels).map(DynamicImage::ImageRgb8),
        Some(ColorSpace::RGBA) => RgbaImage::from_raw(width, height, pixels).map(DynamicImage::ImageRgba8),
        Some(ColorSpace::Luma) => GrayImage::from_raw(width, height, pixels).map(DynamicImage::ImageLuma8),
        Some(ColorSpace::LumaA) => GrayAlphaImage::from_raw(width, height, pixels).map(DynamicImage::ImageLumaA8),
        _ => None,
    };
    image.map(DynamicImage::into_rgb8).ok_or(VisionPreprocessingError::HeicPixelFormat(colorspace))
}

fn get_new_size(
    image: &RgbImage,
    factor_size: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), VisionPreprocessingError> {
    let (width, height) = (f64::from(image.width()), f64::from(image.height()));
    let max_size = width.max(height);
    let min_size = width.min(height);
    if max_size / min_size > MAX_ASPECT_RATIO {
        return Err(VisionPreprocessingError::AspectRatio {
            width: image.width(),
            height: image.height(),
            max_ratio: MAX_ASPECT_RATIO,
        });
    }

    let factor_size = factor_size as f64;
    let min_pixels = min_pixels as f64;
    let max_pixels = max_pixels as f64;
    let mut new_width = (width / factor_size).round_ties_even() * factor_size;
    let mut new_height = (height / factor_size).round_ties_even() * factor_size;

    if new_width * new_height > max_pixels {
        let scale = (width * height / max_pixels).sqrt();
        new_width = factor_size.max((width / scale / factor_size).floor() * factor_size);
        new_height = factor_size.max((height / scale / factor_size).floor() * factor_size);
    } else if new_width * new_height < min_pixels {
        let scale = (min_pixels / (width * height)).sqrt();
        new_width = (width * scale / factor_size).ceil() * factor_size;
        new_height = (height * scale / factor_size).ceil() * factor_size;
    }

    Ok((new_width as usize, new_height as usize))
}

/// Convert RGB image into normalized representation.
/// For each channel new value is:
///     norm = ((pixel_ch / 255) - mean_ch) / std_ch
fn normalize(
    image: &RgbImage,
    mean: [f32; 3],
    std: [f32; 3],
) -> Vec<f32> {
    let mut normalized = Vec::with_capacity(image.as_raw().len());
    for pixel in image.pixels() {
        for channel in 0..3 {
            let norm = (f32::from(pixel.0[channel]) / 255.0 - mean[channel]) / std[channel];
            normalized.push(norm);
        }
    }
    normalized
}

/// Splits a normalized image into patch rows for the patch embedding.
///
/// `normalized` is the interleaved RGB output of [`normalize`].
/// `width` and `height` must be multiples of `patch_size * merge_size` (guaranteed by [`get_new_size`]),
/// otherwise trailing patches are dropped.
///
/// Example: 64x64 image, `patch_size = 16`, `merge_size = 2`.
///
/// 1. Split into 16x16 patches:
///    A  B  C  D
///    E  F  G  H
///    I  J  K  L
///    M  N  O  P
///
/// 2. Order patches by merge window, so each 2x2 group that the merger turns into one LLM
///    token is contiguous:
///    [A B E F] [C D G H] [I J M N] [K L O P]
///
/// 3. Lay out each patch as `[channel, frame, y, x]`, matching the Conv3d weight:
///    3 channels x `temporal_patch_size` frames x 16x16 pixels, i.e. 1536 values for Qwen.
///    A still image has one frame, so the other frames are copies of it.
///
/// Returns `[grid_height * grid_width, 3 * temporal_patch_size * patch_size * patch_size]`, row-major.
fn patchify(
    normalized: &[f32],
    width: usize,
    height: usize,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
) -> Vec<f32> {
    let grid_width = width / patch_size;
    let grid_height = height / patch_size;
    let frame_size = patch_size * patch_size;

    let mut patches = Vec::with_capacity(grid_width * grid_height * 3 * temporal_patch_size * frame_size);
    for block_row in 0..grid_height / merge_size {
        for block_col in 0..grid_width / merge_size {
            for merge_row in 0..merge_size {
                for merge_col in 0..merge_size {
                    let top = (block_row * merge_size + merge_row) * patch_size;
                    let left = (block_col * merge_size + merge_col) * patch_size;
                    for channel in 0..3 {
                        let frame_start = patches.len();

                        // t = 0: this channel's patch_size × patch_size block
                        for y in top..top + patch_size {
                            for x in left..left + patch_size {
                                let value = normalized[(y * width + x) * 3 + channel];
                                patches.push(value);
                            }
                        }

                        // Temporal padding (Qwen's VL format peculiarities):
                        // t = 1..temporal_patch_size repeats the last (only) frame
                        for _ in 1..temporal_patch_size {
                            patches.extend_from_within(frame_start..frame_start + frame_size);
                        }
                    }
                }
            }
        }
    }

    patches
}
