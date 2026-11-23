use std::io::Cursor;

use image::{
    DynamicImage, GrayAlphaImage, GrayImage, ImageDecoder, ImageError, ImageReader, RgbImage, RgbaImage, imageops,
    imageops::FilterType,
};
use shoji::types::session::vision::VisionImage;
use zune_heic::{
    HeicErrors, HeifDecoder,
    zune_core::{bytestream::ZCursor, colorspace::ColorSpace},
};

const MAX_ASPECT_RATIO: f64 = 200.0;

#[derive(Debug, thiserror::Error)]
pub enum VisionScalingError {
    #[error("Aspect ratio too large: {width}x{height}, max ratio: {max_ratio}")]
    AspectRatio {
        width: u32,
        height: u32,
        max_ratio: f64,
    },
    #[error(transparent)]
    Heic(#[from] HeicErrors),
    #[error("Unsupported HEIC pixel format: {0:?}")]
    HeicPixelFormat(Option<ColorSpace>),
    #[error(transparent)]
    Image(#[from] ImageError),
}

/// Decodes an image and scales it as HF `smart_resize` does: both sides become multiples of `factor`
/// (`patch_size * spatial_merge_size`) and the area fits in `min_pixels..=max_pixels`, keeping the aspect ratio.
pub fn scale(
    image_bytes: &[u8],
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<VisionImage, VisionScalingError> {
    let image = get_rgb_image(image_bytes)?;
    let (width, height) = get_new_size(&image, factor, min_pixels, max_pixels)?;
    let image = imageops::resize(&image, width, height, FilterType::CatmullRom);
    Ok(VisionImage {
        pixels: image.into_raw().into_boxed_slice(),
        width,
        height,
    })
}

fn get_rgb_image(image_bytes: &[u8]) -> Result<RgbImage, VisionScalingError> {
    let reader = ImageReader::new(Cursor::new(image_bytes)).with_guessed_format().map_err(ImageError::from)?;
    if reader.format().is_none() && image_bytes.get(4..8) == Some(b"ftyp".as_slice()) {
        return get_heic_rgb_image(image_bytes);
    }

    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    Ok(image.into_rgb8())
}

/// HEIC orientation (`irot`/`imir`) is applied by the decoder.
fn get_heic_rgb_image(image_bytes: &[u8]) -> Result<RgbImage, VisionScalingError> {
    let mut decoder = HeifDecoder::new(ZCursor::new(image_bytes));
    decoder.decode_headers()?;
    let output_bytes = decoder.output_buffer_size().ok_or_else(|| {
        ImageError::Limits(image::error::LimitError::from_kind(image::error::LimitErrorKind::InsufficientMemory))
    })?;
    image::Limits::default().reserve_usize(output_bytes)?;
    let pixels = decoder.decode()?;
    let colorspace = decoder.colorspace();
    let (Some(width), Some(height)) = (decoder.width(), decoder.height()) else {
        return Err(VisionScalingError::HeicPixelFormat(colorspace));
    };
    let (width, height) = (width as u32, height as u32);
    let image = match colorspace {
        Some(ColorSpace::RGB) => RgbImage::from_raw(width, height, pixels).map(DynamicImage::ImageRgb8),
        Some(ColorSpace::RGBA) => RgbaImage::from_raw(width, height, pixels).map(DynamicImage::ImageRgba8),
        Some(ColorSpace::Luma) => GrayImage::from_raw(width, height, pixels).map(DynamicImage::ImageLuma8),
        Some(ColorSpace::LumaA) => GrayAlphaImage::from_raw(width, height, pixels).map(DynamicImage::ImageLumaA8),
        _ => None,
    };
    image.map(DynamicImage::into_rgb8).ok_or(VisionScalingError::HeicPixelFormat(colorspace))
}

fn get_new_size(
    image: &RgbImage,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32), VisionScalingError> {
    let (width, height) = (f64::from(image.width()), f64::from(image.height()));
    if width.max(height) / width.min(height) > MAX_ASPECT_RATIO {
        return Err(VisionScalingError::AspectRatio {
            width: image.width(),
            height: image.height(),
            max_ratio: MAX_ASPECT_RATIO,
        });
    }

    let factor = f64::from(factor);
    let min_pixels = f64::from(min_pixels);
    let max_pixels = f64::from(max_pixels);
    let mut new_width = (width / factor).round_ties_even() * factor;
    let mut new_height = (height / factor).round_ties_even() * factor;
    if new_width * new_height > max_pixels {
        let scale = (width * height / max_pixels).sqrt();
        new_width = factor.max((width / scale / factor).floor() * factor);
        new_height = factor.max((height / scale / factor).floor() * factor);
    } else if new_width * new_height < min_pixels {
        let scale = (min_pixels / (width * height)).sqrt();
        new_width = (width * scale / factor).ceil() * factor;
        new_height = (height * scale / factor).ceil() * factor;
    }

    Ok((new_width as u32, new_height as u32))
}
