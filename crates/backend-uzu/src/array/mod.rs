mod array;
mod context_ext;
mod dense;
mod util;

pub use array::Array;
pub use context_ext::ArrayContextExt;
pub use dense::{
    AllocationAccessError, allocation_as_bytes, allocation_as_bytes_mut, allocation_copy_from_slice, allocation_to_vec,
    try_allocation_to_vec,
};
pub use util::size_for_shape;
