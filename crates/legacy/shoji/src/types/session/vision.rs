/// RGB image already scaled to the vision encoder grid
pub struct VisionImage {
    pub pixels: Box<[u8]>,
    pub width: u32,
    pub height: u32,
}
