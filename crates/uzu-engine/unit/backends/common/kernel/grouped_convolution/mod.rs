#[cfg(backend = "metal")]
mod bench;
mod test;

#[derive(Clone, Copy)]
struct ConvolutionShape {
    sequence_length: usize,
    model_dim: usize,
    group_size: usize,
    kernel_size: usize,
}

impl ConvolutionShape {
    fn groups(self) -> usize {
        self.model_dim / self.group_size
    }

    fn input_len(self) -> usize {
        self.sequence_length * self.model_dim
    }

    fn coefficients_len(self) -> usize {
        self.sequence_length * 2 * self.kernel_size * self.groups()
    }

    fn base_kernel_len(self) -> usize {
        2 * self.kernel_size * self.model_dim
    }

    fn stage_offsets(
        self,
        stage: usize,
        element_size: usize,
    ) -> (usize, usize) {
        (
            stage * self.kernel_size * self.groups() * element_size,
            stage * self.kernel_size * self.model_dim * element_size,
        )
    }
}

const INPUT_STAGE: usize = 0;
const OUTPUT_STAGE: usize = 1;
