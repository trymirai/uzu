use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{Backend, Context, SparseBuffer},
        metal::{Metal, error::MetalError},
    },
    tests::helpers::create_context,
};

#[uzu_test]
fn test_metal_sparse_buffer_capacity_boundary() {
    let context = create_context::<Metal>();
    let capacity = 64 * 1024;
    let mut buffer = context.create_sparse_buffer(capacity).unwrap();

    // 1. Zero length mapping should succeed
    assert!(buffer.map(0).is_ok());

    // 2. Progressive mapping within capacity should succeed
    assert!(buffer.map(16 * 1024).is_ok());
    assert!(buffer.map(32 * 1024).is_ok());
    assert!(buffer.map(capacity).is_ok());

    // 3. Mapping beyond capacity must return CapacityExceeded instead of triggering GPU driver crash
    let err = buffer.map(capacity + 1).unwrap_err();
    match err {
        MetalError::CapacityExceeded {
            requested,
            capacity: cap,
        } => {
            assert_eq!(requested, capacity + 1);
            assert_eq!(cap, capacity);
        },
        other => panic!("expected CapacityExceeded, got {other:?}"),
    }

    // 4. Significantly large out-of-bounds requests are also intercepted safely
    let large_err = buffer.map(capacity * 10).unwrap_err();
    match large_err {
        MetalError::CapacityExceeded {
            requested,
            capacity: cap,
        } => {
            assert_eq!(requested, capacity * 10);
            assert_eq!(cap, capacity);
        },
        other => panic!("expected CapacityExceeded, got {other:?}"),
    }

    // 5. After failed maps, buffer state remains valid and idempotent re-map succeeds
    assert!(buffer.map(capacity).is_ok());
    assert!(buffer.map(32 * 1024).is_ok());
}
