use crate::backends::{
    common::{Backend, Context, SparseBuffer},
    metal::Metal,
};

#[test]
fn test() {
    let ctx = <Metal as Backend>::Context::new().expect("Failed to create Metal context");
    let mut sparse_buffer = ctx.create_sparse_buffer(1024 * 1024).expect("Failed to create sparse buffer");
    let range = 0..4;
    sparse_buffer.mapping(ctx.as_ref(), &range).expect("Failed to map sparse buffer");
}
