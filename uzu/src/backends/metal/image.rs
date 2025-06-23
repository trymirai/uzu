use std::fmt;

use metal::{
    Device, MTLPixelFormat, MTLStorageMode, MTLTextureUsage, Texture,
    TextureDescriptor, TextureRef,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    RGBA8Unorm,
    RGBA32Float,
    BGRA8Unorm,
    R8Unorm,
    R32Float,
}

impl From<PixelFormat> for MTLPixelFormat {
    fn from(format: PixelFormat) -> Self {
        match format {
            PixelFormat::RGBA8Unorm => MTLPixelFormat::RGBA8Unorm,
            PixelFormat::RGBA32Float => MTLPixelFormat::RGBA32Float,
            PixelFormat::BGRA8Unorm => MTLPixelFormat::BGRA8Unorm,
            PixelFormat::R8Unorm => MTLPixelFormat::R8Unorm,
            PixelFormat::R32Float => MTLPixelFormat::R32Float,
        }
    }
}

impl From<MTLPixelFormat> for PixelFormat {
    fn from(format: MTLPixelFormat) -> Self {
        match format {
            MTLPixelFormat::RGBA8Unorm => PixelFormat::RGBA8Unorm,
            MTLPixelFormat::RGBA32Float => PixelFormat::RGBA32Float,
            MTLPixelFormat::BGRA8Unorm => PixelFormat::BGRA8Unorm,
            MTLPixelFormat::R8Unorm => PixelFormat::R8Unorm,
            MTLPixelFormat::R32Float => PixelFormat::R32Float,
            _ => panic!("Unsupported pixel format conversion"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureUsage {
    pub shader_read: bool,
    pub shader_write: bool,
    pub render_target: bool,
    pub pixel_format_view: bool,
}

impl Default for TextureUsage {
    fn default() -> Self {
        Self {
            shader_read: true,
            shader_write: true,
            render_target: false,
            pixel_format_view: false,
        }
    }
}

impl From<TextureUsage> for MTLTextureUsage {
    fn from(usage: TextureUsage) -> Self {
        let mut result = MTLTextureUsage::empty();
        if usage.shader_read {
            result = result | MTLTextureUsage::ShaderRead;
        }
        if usage.shader_write {
            result = result | MTLTextureUsage::ShaderWrite;
        }
        if usage.render_target {
            result = result | MTLTextureUsage::RenderTarget;
        }
        if usage.pixel_format_view {
            result = result | MTLTextureUsage::PixelFormatView;
        }
        result
    }
}

#[derive(Debug)]
pub struct Image {
    texture: Texture,
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
}

impl Image {
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
    ) -> Self {
        Self::new_with_usage(
            device,
            width,
            height,
            pixel_format,
            TextureUsage::default(),
        )
    }

    pub fn new_with_usage(
        device: &Device,
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        usage: TextureUsage,
    ) -> Self {
        let descriptor = TextureDescriptor::new();
        descriptor.set_pixel_format(pixel_format.into());
        descriptor.set_width(width as u64);
        descriptor.set_height(height as u64);
        descriptor.set_storage_mode(MTLStorageMode::Private);
        descriptor.set_usage(usage.into());

        let texture = device.new_texture(&descriptor);
        texture.set_label("Image");

        Self {
            texture,
            width,
            height,
            pixel_format,
        }
    }

    pub fn new_with_descriptor(
        device: &Device,
        descriptor: &TextureDescriptor,
    ) -> Self {
        let texture = device.new_texture(descriptor);
        texture.set_label("Image");

        let width = descriptor.width() as u32;
        let height = descriptor.height() as u32;
        let pixel_format = descriptor.pixel_format().into();

        Self {
            texture,
            width,
            height,
            pixel_format,
        }
    }

    pub fn from_texture(texture: Texture) -> Self {
        let width = texture.width() as u32;
        let height = texture.height() as u32;
        let pixel_format = texture.pixel_format().into();

        Self {
            texture,
            width,
            height,
            pixel_format,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }

    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    pub fn texture_ref(&self) -> &TextureRef {
        self.texture.as_ref()
    }

    pub fn set_label(
        &self,
        label: &str,
    ) {
        self.texture.set_label(label);
    }
}

impl AsRef<TextureRef> for Image {
    fn as_ref(&self) -> &TextureRef {
        self.texture.as_ref()
    }
}

impl fmt::Display for Image {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "Image({}x{}, {:?})",
            self.width, self.height, self.pixel_format
        )
    }
}
