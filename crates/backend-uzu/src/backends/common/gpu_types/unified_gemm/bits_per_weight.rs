use debug_display::Display;

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitsPerWeight {
    Bits4 = 4,
    Bits8 = 8,
}

impl BitsPerWeight {
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    pub const fn weights_per_byte(self) -> u32 {
        8 / self.as_u32()
    }

    pub const fn packed_bytes_for_k(self, k: u32) -> u32 {
        k / self.weights_per_byte()
    }
}
