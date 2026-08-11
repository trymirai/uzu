use crate::{
    backends::common::{Allocation, Backend},
    data_type::DataType,
};

pub struct Array<B: Backend> {
    path: String,
    shape: Box<[usize]>,
    data_type: DataType,
    allocation: Allocation<B>,
}
