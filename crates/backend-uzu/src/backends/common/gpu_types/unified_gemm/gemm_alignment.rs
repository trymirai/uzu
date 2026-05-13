use bitflags::bitflags;

bitflags! {
    /// Which GEMM axes are evenly divisible by the tile shape. Passed to the
    /// kernel through a `uint32_t` function-constant slot whose bits the
    /// Metal side tests directly — there is no corresponding GPU struct.
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GemmAlignment: u32 {
        const M = 1 << 0;
        const N = 1 << 1;
        const K = 1 << 2;
    }
}

impl GemmAlignment {
    pub fn from_flags(
        m: bool,
        n: bool,
        k: bool,
    ) -> Self {
        let mut bits = Self::empty();
        if m {
            bits |= Self::M;
        }
        if n {
            bits |= Self::N;
        }
        if k {
            bits |= Self::K;
        }
        bits
    }

    pub fn m_aligned(self) -> bool {
        self.contains(Self::M)
    }

    pub fn n_aligned(self) -> bool {
        self.contains(Self::N)
    }

    pub fn k_aligned(self) -> bool {
        self.contains(Self::K)
    }
}
