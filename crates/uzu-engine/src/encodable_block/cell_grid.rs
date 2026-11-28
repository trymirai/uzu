/// Cells (vision patches or image tokens) in raster order of `merge_size`×`merge_size` blocks
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellGrid {
    pub width: u32,
    pub height: u32,
    pub merge_size: u32,
}

impl CellGrid {
    pub fn new(
        width: u32,
        height: u32,
        merge_size: u32,
    ) -> Self {
        Self {
            width,
            height,
            merge_size,
        }
    }

    pub fn cell_count(self) -> u32 {
        self.width * self.height
    }

    pub fn merged(self) -> Self {
        Self::new(self.width / self.merge_size, self.height / self.merge_size, 1)
    }
}
