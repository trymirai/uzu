use proc_macros::uzu_test;

use crate::tests::cold_pool::copy_count;

#[uzu_test]
fn copy_count_math() {
    assert_eq!(copy_count(512 << 20, 64 << 20), 8);
    assert_eq!(copy_count(512 << 20, 100 << 20), 6);
    assert_eq!(copy_count(512 << 20, 1 << 30), 1);
}
