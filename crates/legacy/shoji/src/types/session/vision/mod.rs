/// RGB image already scaled to the vision encoder grid
pub struct VisionImage {
    pub pixels: Box<[u8]>,
    pub width: u32,
    pub height: u32,
}

/// An image and the place of its `<|image_pad|>` token in the stream input tokens
pub struct VisionImageInput {
    /// Index of the image's pad token in the input tokens
    pub start: usize,
    pub image: VisionImage,
}
