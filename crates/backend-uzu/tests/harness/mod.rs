#![cfg(test)]

mod uzu_test;

pub extern crate test;

pub use uzu_test::UzuTest;

pub fn uzu_harness(tests: &[&UzuTest]) {
    let tests = tests
        .iter()
        .filter_map(|test| match test {
            UzuTest::Test(test) => Some(*test),
            UzuTest::Bench(_) => None,
        })
        .collect::<Vec<_>>();
    test::test_main_static(&tests)
}
