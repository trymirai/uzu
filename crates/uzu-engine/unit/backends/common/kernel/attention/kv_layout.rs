use uzu_engine_macros::uzu_test;

pub use crate::backends::common::kernel::kv_layout::Int8KVLayout;

#[uzu_test]
fn code_planes_are_bijections() {
    let layout = Int8KVLayout::new(256, 2, 96);
    for index in [Int8KVLayout::k_code_index, Int8KVLayout::v_code_index] {
        let mut bytes: Vec<usize> = (0..96 * 2 * 256).map(|i| index(&layout, i / 512, i / 256 % 2, i % 256)).collect();
        bytes.sort_unstable();
        assert!(bytes.into_iter().eq(0..layout.code_bytes()));
    }
}
