use num_traits::Float;

/// One raw short convolution problem shared by the four kernels' CPU and Vulkan runners: dimensions and every payload.
/// `state` is Pack's state_in, Decode's state and Trie's base_state; `next_state` the initial Prefill state_out and
/// out-of-place Decode next_state, so elements past the copied taps must survive.
#[derive(Clone)]
pub struct ShortConvCase<T> {
    pub model_dim: u32,
    pub kernel_size: u32,
    pub suffix_len: u32,
    pub state_stride: u32,
    pub in_proj_stride: u32,
    /// [suffix_len, in_proj_stride]: pre-gate, post-gate and input at 0, model_dim and 2 model_dim.
    pub in_proj: Vec<T>,
    /// [model_dim, kernel_size].
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
    /// [model_dim, state_stride].
    pub state: Vec<T>,
    pub next_state: Vec<T>,
    /// [state_stride + suffix_len, model_dim], Prefill's input.
    pub padded: Vec<T>,
    /// Trie parents, a chain by default.
    pub parents: Vec<i32>,
}

impl<T: Float> ShortConvCase<T> {
    pub fn new(
        (model_dim, kernel_size, suffix_len, state_stride, in_proj_stride): (u32, u32, u32, u32, u32),
        seed: usize,
    ) -> Self {
        assert!(kernel_size > 0 && state_stride + 1 >= kernel_size && in_proj_stride >= 3 * model_dim);
        let (dim, suffix, stride) = (model_dim as usize, suffix_len as usize, state_stride as usize);
        Self {
            model_dim,
            kernel_size,
            suffix_len,
            state_stride,
            in_proj_stride,
            in_proj: Self::values(suffix * in_proj_stride as usize, seed),
            w: ShortConvCase::<f32>::values(dim * kernel_size as usize, seed + 1),
            b: None,
            state: Self::values(dim * stride, seed + 2),
            next_state: Self::values(dim * stride, seed + 3),
            padded: Self::values((stride + suffix) * dim, seed + 4),
            parents: (-1..suffix as i32 - 1).collect(),
        }
    }

    /// Signed values with magnitudes in [1/8, 2], so FP32 products of two stay normal in every storage type.
    pub fn values(
        length: usize,
        seed: usize,
    ) -> Vec<T> {
        (0..length)
            .map(|i| {
                let k = i * 7919 + seed * 104_729;
                let magnitude = 0.125 + (k % 1021) as f32 / 1020.0 * 1.875;
                T::from(if (k / 1021).is_multiple_of(3) {
                    -magnitude
                } else {
                    magnitude
                })
                .unwrap()
            })
            .collect()
    }

    pub fn bias(mut self) -> Self {
        self.b = Some(ShortConvCase::<f32>::values(self.model_dim as usize, 5));
        self
    }

    pub fn parents(
        mut self,
        parents: &[i32],
    ) -> Self {
        assert!(parents.len() == self.suffix_len as usize);
        assert!(parents.iter().enumerate().all(|(node, &parent)| parent < node as i32));
        self.parents = parents.to_vec();
        self
    }

    pub fn label(&self) -> String {
        format!(
            "dim {} kernel {} suffix {} state_stride {} in_proj_stride {}{}",
            self.model_dim,
            self.kernel_size,
            self.suffix_len,
            self.state_stride,
            self.in_proj_stride,
            if self.b.is_some() {
                " bias"
            } else {
                ""
            }
        )
    }
}
