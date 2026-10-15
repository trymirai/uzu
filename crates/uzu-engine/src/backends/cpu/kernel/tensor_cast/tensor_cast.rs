use half::bf16;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

#[kernel(TensorCast)]
#[variants(InputT, bf16)]
#[variants(OutputT, f32)]
pub fn tensor_cast<InputT: DataTypeRepr + num_traits::ToPrimitive, OutputT: DataTypeRepr + num_traits::NumCast>(
    input: *const InputT,
    output: *mut OutputT,
    length: u32,
) {
    for index in 0..length as usize {
        unsafe {
            *output.add(index) = OutputT::from((*input.add(index)).to_f32().unwrap()).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use uzu_engine_macros::uzu_test;

    use super::tensor_cast;

    #[uzu_test]
    fn d3_bf16_to_f32_cast_preserves_values() {
        let input = [-3.125, -0.0, 0.0, 0.03125, 1.0, 128.0].map(bf16::from_f32);
        let mut output = [f32::NAN; 6];
        tensor_cast(input.as_ptr(), output.as_mut_ptr(), input.len() as u32);
        for (input, output) in input.iter().zip(output) {
            assert_eq!(input.to_f32().to_bits(), output.to_bits());
        }
    }
}
