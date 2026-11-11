use std::io::Cursor;

use image::{DynamicImage, ImageDecoder, ImageError, ImageReader, RgbImage, imageops, imageops::FilterType};

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
}

pub struct VisionPreprocessingConfig {
    pub patch_size: usize,
    pub merge_size: usize,
    pub shortest_edge: usize,
    pub longest_edge: usize,
}

pub struct VisionPreprocessingOutput {}

pub fn preprocess(
    image_bytes: &[u8],
    config: &VisionPreprocessingConfig,
) -> Result<VisionPreprocessingOutput, VisionPreprocessingError> {
    let image = get_rgb_image(image_bytes).map_err(VisionPreprocessingError::Image)?;
    let factor_size = config.merge_size * config.patch_size;
    let (new_width, new_height) = get_new_size(&image, factor_size, config.shortest_edge, config.longest_edge)?;
    let image = imageops::resize(&image, new_width, new_height, FilterType::CatmullRom);
    Ok(VisionPreprocessingOutput {})
}

fn get_new_size(
    image: &RgbImage,
    factor_size: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(u32, u32), VisionPreprocessingError> {
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

    Ok((new_width as u32, new_height as u32))
}

fn get_rgb_image(image_bytes: &[u8]) -> Result<RgbImage, ImageError> {
    let cursor = Cursor::new(image_bytes);
    let reader = ImageReader::new(cursor).with_guessed_format().map_err(ImageError::from)?;
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image.into_rgb8())
}
