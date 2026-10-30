use derive_more::Display;

/// A Gemm workgroup block, M x N x K, and its subgroups along M, N and K; tensor tilings run the cooperative core.
#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GemmTiling {
    Tile8x32x32_Simdgroups1x1,
    Tile64x32x32_Simdgroups2x2,
    Tile64x64x16_Simdgroups2x2,
    Tile64x64x32_Simdgroups2x2,
    Tile32x32x32_Simdgroups2x2,
    Tile16x64x64_Simdgroups1x4x2,
    Tile32x64x64_Simdgroups1x4x2,
    Tile64x128x32_Simdgroups2x4x1,
    Tile128x128x32_Simdgroups2x4x1,
}

impl GemmTiling {
    const fn shape(self) -> [u32; 6] {
        match self {
            Self::Tile8x32x32_Simdgroups1x1 => [8, 32, 32, 1, 1, 1],
            Self::Tile64x32x32_Simdgroups2x2 => [64, 32, 32, 2, 2, 1],
            Self::Tile64x64x16_Simdgroups2x2 => [64, 64, 16, 2, 2, 1],
            Self::Tile64x64x32_Simdgroups2x2 => [64, 64, 32, 2, 2, 1],
            Self::Tile32x32x32_Simdgroups2x2 => [32, 32, 32, 2, 2, 1],
            Self::Tile16x64x64_Simdgroups1x4x2 => [16, 64, 64, 1, 4, 2],
            Self::Tile32x64x64_Simdgroups1x4x2 => [32, 64, 64, 1, 4, 2],
            Self::Tile64x128x32_Simdgroups2x4x1 => [64, 128, 32, 2, 4, 1],
            Self::Tile128x128x32_Simdgroups2x4x1 => [128, 128, 32, 2, 4, 1],
        }
    }

    pub const fn block_m(self) -> u32 {
        self.shape()[0]
    }

    pub const fn block_n(self) -> u32 {
        self.shape()[1]
    }

    pub const fn block_k(self) -> u32 {
        self.shape()[2]
    }

    pub const fn simdgroups(self) -> u32 {
        let [.., m, n, k] = self.shape();
        m * n * k
    }

    pub const fn is_tensor(self) -> bool {
        self as u32 >= Self::Tile16x64x64_Simdgroups1x4x2 as u32
    }

    /// Threadgroup elements of the scalar core's A (`left`) or B staging, rows padded by 16 bytes of `element_bytes`.
    pub const fn scalar_staging(
        self,
        left: bool,
        element_bytes: u32,
    ) -> u32 {
        let rows = if left {
            self.block_m()
        } else {
            self.block_n()
        };
        rows * (self.block_k() + 16 / element_bytes)
    }

    /// 16-byte words of the cooperative core's two stages of BF16 A and B, rows padded by 8 values.
    pub const fn tensor_staging(self) -> u32 {
        2 * (self.block_m() + self.block_n()) * (self.block_k() + 8) / 8
    }
}

/// Int8Gemm's fold metadata entries per array, [stage parity * 2 + step][kind][row or column].
pub const INT8_FOLD_ENTRIES: u32 = 4 * 2 * 64;

/// The most input rows a GEMV workgroup takes at once.
pub const GEMV_INPUT_ROW_TILE: u32 = 16;
