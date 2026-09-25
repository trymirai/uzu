use crate::backends::common::{Allocation, Backend, Encoder, Kernels, kernel::Unsupported};

/// Trellis code layouts of Mirai S linear weights: V4 restarts its state every 64 columns, V2 runs one trellis over
/// the whole row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrellisCodec {
    V4,
    V2T6,
    V2T4,
}

impl TrellisCodec {
    /// (columns per 16-bit state, bits per transition)
    pub fn shape(self) -> (u32, u32) {
        match self {
            Self::V4 => (4, 8),
            Self::V2T6 => (2, 6),
            Self::V2T4 => (2, 4),
        }
    }

    pub fn row_bytes(
        self,
        columns: u32,
    ) -> u32 {
        match self {
            Self::V4 => columns / 64 * 17,
            Self::V2T6 | Self::V2T4 => (16 + (columns / 2 - 1) * self.shape().1).div_ceil(8),
        }
    }
}

/// Order of the mixing block matrix of a `columns`-wide input rotation (its odd part; the rest is Walsh-Hadamard).
pub fn mixing_order(columns: u32) -> u32 {
    columns >> columns.trailing_zeros()
}

/// A linear's input after the Mirai S rotation, quantized to int8 per token.
pub struct RotatedInput<B: Backend> {
    /// i8 `[batch, columns]`, rows past `batch` padded up to the projection's token tile.
    pub activations: Allocation<B>,
    /// f32 `[batch, 8]`: the int8 sums over the columns k = j (mod 4) for j in 0..4, the activation scale, 0, 0, 0.
    pub token_statistics: Allocation<B>,
    pub batch: u32,
    pub columns: u32,
}

pub trait MiraiSTransform: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<MiraiSTransform = Self>>;

    /// `None` when the device cannot run Mirai S kernels or has no transform for `columns`.
    fn new(
        context: &<Self::Backend as Backend>::Context,
        columns: u32,
    ) -> Result<Option<Self>, <Self::Backend as Backend>::Error>;

    /// Rotates `input` (bf16 `[batch, columns]`) by `signs`, the `mixing` block matrix and a Walsh-Hadamard
    /// transform, then quantizes each token to int8.
    fn encode(
        &self,
        input: &Allocation<Self::Backend>,
        signs: &Allocation<Self::Backend>,
        mixing: &Allocation<Self::Backend>,
        batch: u32,
        encoder: &mut Encoder<Self::Backend>,
    ) -> Result<RotatedInput<Self::Backend>, <Self::Backend as Backend>::Error>;
}

pub struct ProjectionArguments<'a, B: Backend> {
    pub input: &'a RotatedInput<B>,
    pub codes: &'a Allocation<B>,
    pub row_scales: &'a Allocation<B>,
    pub codebook: &'a Allocation<B>,
    pub rows: u32,
    /// Rows `output_row_offset..` of the `[batch, output_stride]` output.
    pub output: &'a mut Allocation<B>,
    pub output_row_offset: u32,
    pub output_stride: u32,
}

pub trait MiraiSProjection: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<MiraiSProjection = Self>>;

    /// `None` when the device cannot run Mirai S kernels.
    fn new(
        context: &<Self::Backend as Backend>::Context,
        codec: TrellisCodec,
    ) -> Result<Option<Self>, <Self::Backend as Backend>::Error>;

    fn encode(
        &self,
        arguments: ProjectionArguments<'_, Self::Backend>,
        encoder: &mut Encoder<Self::Backend>,
    );
}

impl<B: Backend<Kernels: Kernels<MiraiSTransform = Unsupported<B>>>> MiraiSTransform for Unsupported<B> {
    type Backend = B;

    fn new(
        _context: &B::Context,
        _columns: u32,
    ) -> Result<Option<Self>, B::Error> {
        Ok(None)
    }

    fn encode(
        &self,
        _input: &Allocation<B>,
        _signs: &Allocation<B>,
        _mixing: &Allocation<B>,
        _batch: u32,
        _encoder: &mut Encoder<B>,
    ) -> Result<RotatedInput<B>, B::Error> {
        match self.never {}
    }
}

impl<B: Backend<Kernels: Kernels<MiraiSProjection = Unsupported<B>>>> MiraiSProjection for Unsupported<B> {
    type Backend = B;

    fn new(
        _context: &B::Context,
        _codec: TrellisCodec,
    ) -> Result<Option<Self>, B::Error> {
        Ok(None)
    }

    fn encode(
        &self,
        _arguments: ProjectionArguments<'_, B>,
        _encoder: &mut Encoder<B>,
    ) {
        match self.never {}
    }
}
