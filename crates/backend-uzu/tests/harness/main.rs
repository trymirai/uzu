#[cfg(test)]
extern crate test;

#[cfg(test)]
pub fn uzu_harness(tests: &[&test::TestDescAndFn]) {
    println!("Running uzu-harness {}", tests.len());
    test::test_main_static(tests)
}
