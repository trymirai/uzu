use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder,
        kernel::{Kernels, NormalizationKernel},
    },
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
    parameters::{ParameterLoaderError, ParameterTree},
};

pub struct Recorder<B: Backend> {
    arrays: Vec<Allocation<B>>,
}
