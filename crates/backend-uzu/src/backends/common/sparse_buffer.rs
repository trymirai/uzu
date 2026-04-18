use std::{cell::RefCell, fmt::Debug, rc::Rc};

use crate::backends::common::Backend;

pub trait SparseBuffer: Debug {
    type Backend: Backend<SparseBuffer = Self>;

    fn buffer(&self) -> Rc<RefCell<<Self::Backend as Backend>::Buffer>>;

    fn capacity(&self) -> usize;

    fn extend(
        &mut self,
        add_length: usize,
    );

    fn length(&self) -> usize;
}
