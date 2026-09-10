use crate::{
    array::size_for_shape,
    backends::common::{Allocation, Backend, Encoder, Kernels, kernel::TransposeKernel},
    data_type::DataType,
};

pub struct Transpose<B: Backend> {
    kernel: <B::Kernels as Kernels>::TransposeKernel,
    data_type: DataType,
    in_place: bool,
}

impl<B: Backend> Transpose<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        in_place: bool,
    ) -> Result<Self, B::Error> {
        let bits = match data_type {
            DataType::U4 | DataType::U8 | DataType::BF16 => data_type.size_in_bits() as u16,
            _ => panic!("transpose supports U4, U8, and BF16 only"),
        };
        Ok(Self {
            kernel: <B::Kernels as Kernels>::TransposeKernel::new(context, bits, in_place)?,
            data_type,
            in_place,
        })
    }

    pub fn encode(
        &self,
        input: &mut Allocation<B>,
        output: Option<&mut Allocation<B>>,
        rows: u32,
        cols: u32,
        encoder: &mut Encoder<B>,
    ) {
        assert!(rows != 0 && cols != 0, "transpose dimensions must be non-zero");
        assert!(!self.in_place || rows == cols, "in-place transpose requires a square matrix");
        assert_eq!(output.is_none(), self.in_place, "transpose mode does not match output");
        let input_size = size_for_shape(&[rows, cols], self.data_type);
        let output_size = size_for_shape(&[cols, rows], self.data_type);
        assert!(
            input.size() >= input_size && output.as_ref().is_none_or(|output| output.size() >= output_size),
            "transpose allocation is too small"
        );
        self.kernel.encode(input, output, rows, cols, encoder);
    }
}
