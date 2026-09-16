use metal::MTLResidencySet;
use uzu_engine_macros::uzu_test;

use crate::backends::{
    common::{Context, DeviceCapabilities},
    metal::MetalContext,
};

#[uzu_test]
fn test_buffers_release_residency() {
    let context = MetalContext::new().unwrap();
    let initial_count = context.residency_set.lock().allocation_count();
    let dense = context.create_buffer(4096).unwrap();
    assert_eq!(context.residency_set.lock().allocation_count(), initial_count + 1);
    drop(dense);
    assert_eq!(context.residency_set.lock().allocation_count(), initial_count);

    if context.device_capabilities().contains(DeviceCapabilities::SPARSE_BUFFERS) {
        let sparse = context.create_sparse_buffer(256 * 1024).unwrap();
        assert_eq!(context.residency_set.lock().allocation_count(), initial_count + 1);
        drop(sparse);
        assert_eq!(context.residency_set.lock().allocation_count(), initial_count);
    }
}
