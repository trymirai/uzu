use super::{
    d_ops::MatmulDOps,
    matmul_a::MatmulA,
    matmul_b::MatmulB,
    routing::{ExpertInput, MatmulRouting},
};
use crate::{
    backends::common::{Allocation, Backend, BufferArg},
    data_type::DataType,
};

pub struct MatmulArguments<'a, 'b, 'd, B: Backend, TB: BufferArg<'b, B> = &'b Allocation<B>> {
    pub a: MatmulA<'a, B>,
    pub b: MatmulB<'b, B, TB>,
    pub b_leading_dimension: Option<u32>,
    pub b_transpose: bool,
    pub d: &'d mut Allocation<B>,
    pub d_transform: MatmulDOps<'d, B>,
    pub routing: MatmulRouting<'a, B>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub(crate) struct MatmulStorageError {
    pub operand: &'static str,
    pub reason: &'static str,
}

pub(crate) fn validate_matmul_storage<'a, 'b, 'd, B: Backend, TB: BufferArg<'b, B>>(
    arguments: &MatmulArguments<'a, 'b, 'd, B, TB>,
    input_data_type: DataType,
    output_data_type: DataType,
) -> Result<(), MatmulStorageError> {
    if let MatmulA::FullPrecision {
        values,
        offset,
    } = &arguments.a
    {
        let element_size = input_data_type.size_in_bytes();
        if !offset.is_multiple_of(element_size) {
            return Err(MatmulStorageError {
                operand: "A",
                reason: "full-precision byte offset is not aligned to the input data type",
            });
        }
        let input_rows = match arguments.routing.expert_routes() {
            Some(routes) if routes.input == ExpertInput::Tokens => arguments.m / routes.routes_per_token.get(),
            _ => arguments.m,
        };
        let required_end = (input_rows as usize)
            .checked_mul(arguments.k as usize)
            .and_then(|size| size.checked_mul(element_size))
            .and_then(|size| size.checked_add(*offset));
        if required_end.is_none_or(|required| values.size() < required) {
            return Err(MatmulStorageError {
                operand: "A",
                reason: "full-precision input allocation does not cover every referenced row",
            });
        }
    }

    let required_output = (arguments.m as usize)
        .checked_mul(arguments.n as usize)
        .and_then(|size| size.checked_mul(output_data_type.size_in_bytes()));
    if required_output.is_none_or(|required| arguments.d.size() < required) {
        return Err(MatmulStorageError {
            operand: "D",
            reason: "output allocation does not cover M * N elements",
        });
    }
    Ok(())
}
