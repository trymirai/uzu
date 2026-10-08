use std::{ops::Range, sync::Arc};

use num_traits::Float;

use super::{NormalizationCase, kernel_fixture::KernelFixture};
use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Kernels, kernel::QKVNormKernel},
        cpu::Cpu,
        vulkan::{QKVNormVulkanKernel, VkBuffer, VkCommandBufferEncoding},
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, submit_command_buffer},
};

/// One raw QKVNorm dispatch shared by the CPU and Vulkan runners: `batch_size` packed rows of `input_row_stride`
/// holding query, key and value heads, a trailing gate and padding, of which `head_count` heads from `head_offset` are
/// normalized; optional scales, the scalars and the in-place mode.
#[derive(Clone)]
pub struct QKVNormCase<I, S> {
    pub input: Vec<I>,
    pub scales: Option<Vec<S>>,
    pub batch_size: u32,
    pub input_row_stride: u32,
    pub head_dim: u32,
    pub epsilon: f32,
    pub scale_offset: f32,
    pub head_offset: u32,
    pub head_count: u32,
    pub full_layer: bool,
    pub in_place: bool,
}

impl<I: ArrayElement + Float, S: ArrayElement + Float> QKVNormCase<I, S> {
    /// Rows of `heads` heads of arbitrary-mantissa values in [-4, 4], then a two-head gate and 3 padding elements;
    /// `seed` varies the pattern.
    pub fn new(
        batch_size: u32,
        heads: u32,
        head_dim: u32,
        (head_offset, head_count): (u32, u32),
        seed: usize,
    ) -> Self {
        let input_row_stride = (heads + 2) * head_dim + 3;
        let length = (batch_size * input_row_stride) as usize;
        Self {
            input: (0..length).map(|i| I::from(((i * 41 + seed) % 613) as f32 / 76.37 - 4.0).unwrap()).collect(),
            scales: None,
            batch_size,
            input_row_stride,
            head_dim,
            epsilon: 1e-5,
            scale_offset: 0.0,
            head_offset,
            head_count,
            full_layer: true,
            in_place: false,
        }
    }

    /// Arbitrary-mantissa scales in [0.5, 1.6] plus `scale_offset`, applied in FP32 when `full_layer` and in the output
    /// type otherwise.
    pub fn scales(
        mut self,
        full_layer: bool,
        scale_offset: f32,
    ) -> Self {
        let count = self.head_dim as usize;
        self.scales = Some((0..count).map(|i| S::from(0.5 + ((i * 13) % 17) as f32 / 15.3).unwrap()).collect());
        (self.full_layer, self.scale_offset) = (full_layer, scale_offset);
        self
    }

    pub fn in_place(mut self) -> Self {
        self.in_place = true;
        self
    }

    /// Flat indices of the normalized elements, head by head of each row.
    pub fn selected(&self) -> Vec<usize> {
        let (stride, head_dim) = (self.input_row_stride as usize, self.head_dim as usize);
        let start = self.head_offset as usize * head_dim;
        let span = self.head_count as usize * head_dim;
        (0..self.batch_size as usize).flat_map(|row| row * stride + start..row * stride + start + span).collect()
    }

    /// The output before the dispatch: the input in place (whose input and output types are equal), otherwise a
    /// distinct pattern that every unselected element must keep.
    pub fn initial_output<O: Float>(&self) -> Vec<O> {
        match self.in_place {
            true => self.input.iter().map(|&value| O::from(value).unwrap()).collect(),
            false => (0..self.input.len()).map(|i| O::from(((i * 7) % 97) as f32 * 0.125 - 6.0).unwrap()).collect(),
        }
    }

    /// The selected heads as rows of a plain RMS Normalization case with the same scales and scalars, whose canonical
    /// stage oracle bounds each head.
    pub fn normalization_case(&self) -> NormalizationCase<I, S> {
        let mut case = NormalizationCase::new(0, self.head_dim, 0);
        case.input = self.selected().into_iter().map(|index| self.input[index]).collect();
        case.batch_size = self.batch_size * self.head_count;
        (case.scales, case.epsilon, case.scale_offset) = (self.scales.clone(), self.epsilon, self.scale_offset);
        case.full_layer = self.full_layer;
        case
    }

    /// The CPU kernel through the shared trait from the initial output; CPU buffers cannot be empty, so empty rows or
    /// scales leave it as is.
    pub fn cpu<O: ArrayElement + Float>(&self) -> Vec<O> {
        let initial = self.initial_output::<O>();
        if initial.is_empty() || self.head_dim == 0 {
            return initial;
        }
        let context = create_context::<Cpu>();
        let kernel = <<Cpu as Backend>::Kernels as Kernels>::QKVNormKernel::new(
            &context,
            I::data_type(),
            S::data_type(),
            O::data_type(),
            DataType::F32,
            self.in_place,
            self.scales.is_some(),
        )
        .expect("CPU QKVNorm");
        let input = (!self.in_place).then(|| create_buffer_with_data::<Cpu, I>(&context, &self.input));
        let scales = self.scales.as_ref().map(|scales| create_buffer_with_data::<Cpu, S>(&context, scales));
        let mut output = create_buffer_with_data::<Cpu, O>(&context, &initial);
        let mut command_buffer = context.create_command_buffer(None, None).expect("CPU command buffer");
        kernel.encode(
            input.as_ref(),
            scales.as_ref(),
            &mut output,
            self.batch_size,
            self.input_row_stride,
            self.head_dim,
            self.epsilon,
            self.scale_offset,
            self.head_offset,
            self.head_count,
            self.full_layer,
            &mut command_buffer,
        );
        submit_command_buffer(command_buffer);
        buffer_to_vec::<Cpu, O>(&output)
    }

    pub fn vulkan_kernel<O: ArrayElement>(
        &self,
        fixture: &KernelFixture,
    ) -> QKVNormVulkanKernel {
        QKVNormVulkanKernel::new(
            &fixture.context,
            I::data_type(),
            S::data_type(),
            O::data_type(),
            DataType::F32,
            self.in_place,
            self.scales.is_some(),
        )
        .expect("Vulkan QKVNorm")
    }

    /// Records every case with `kernel`, built for their shared types and specializations, into `encoding` over guarded
    /// ranges from the initial outputs and completes it. Returns each output after asserting every guard and the input
    /// and scales unchanged.
    pub fn gpu<O: ArrayElement + Float>(
        fixture: &KernelFixture,
        kernel: &QKVNormVulkanKernel,
        cases: &[Self],
        mut encoding: VkCommandBufferEncoding,
    ) -> Vec<Vec<O>> {
        let (input_sentinel, scale_sentinel, sentinel) =
            (I::from(-7.0).unwrap(), S::from(-7.0).unwrap(), O::from(-7.0).unwrap());
        fn range((buffer, range): &(Arc<VkBuffer>, Range<u64>)) -> (&Arc<VkBuffer>, Range<u64>) {
            (buffer, range.clone())
        }
        let buffers = cases
            .iter()
            .map(|case| {
                let input = (!case.in_place).then(|| fixture.guarded(&case.input, input_sentinel));
                let scales = case.scales.as_ref().map(|scales| fixture.guarded(scales, scale_sentinel));
                (input, scales, fixture.guarded(&case.initial_output::<O>(), sentinel))
            })
            .collect::<Vec<_>>();
        for (case, (input, scales, output)) in cases.iter().zip(&buffers) {
            // SAFETY: input and output hold `batch_size` rows of `input_row_stride` and the scales `head_dim`, all
            // aligned; the output aliases nothing, and in place the types are equal.
            unsafe {
                kernel.encode(
                    input.as_ref().map(range),
                    scales.as_ref().map(range),
                    range(output),
                    case.batch_size,
                    case.input_row_stride,
                    case.head_dim,
                    case.epsilon,
                    case.scale_offset,
                    case.head_offset,
                    case.head_count,
                    case.full_layer,
                    &mut encoding,
                );
            }
        }
        KernelFixture::complete(encoding);
        cases
            .iter()
            .zip(&buffers)
            .map(|(case, (input, scales, output))| {
                // SAFETY: the only command buffer using these buffers has completed.
                unsafe {
                    if let Some(input) = input {
                        KernelFixture::assert_unchanged(input, input_sentinel, &case.input, "input");
                    }
                    if let (Some(guarded), Some(payload)) = (scales, &case.scales) {
                        KernelFixture::assert_unchanged(guarded, scale_sentinel, payload, "scales");
                    }
                    KernelFixture::read_guarded(output, sentinel)
                }
            })
            .collect()
    }

    pub fn label<O: ArrayElement>(&self) -> String {
        let scales = match (&self.scales, self.full_layer) {
            (None, _) => "no scales".to_owned(),
            (Some(_), true) => format!("full_layer offset {}", self.scale_offset),
            (Some(_), false) => format!("only_normalization offset {}", self.scale_offset),
        };
        format!(
            "QKVNorm {:?}/{:?}/{:?} {scales}{} batch {} heads {}+{} of {} stride {} epsilon {:e}",
            I::data_type(),
            S::data_type(),
            O::data_type(),
            if self.in_place {
                " in_place"
            } else {
                ""
            },
            self.batch_size,
            self.head_offset,
            self.head_count,
            self.head_dim,
            self.input_row_stride,
            self.epsilon
        )
    }
}
