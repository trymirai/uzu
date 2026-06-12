#![cfg(test)]

mod uzu_test;

pub extern crate test;

use test::TestDescAndFn;
pub use uzu_test::UzuTest;

pub fn uzu_harness(tests: &[&UzuTest]) {
    let args = std::env::args().collect::<Vec<String>>();
    let benchmarks = args.contains(&"--benches".to_string());
    println!("args: {:?}", args);
    if benchmarks {
        let bench_tests: Vec<&dyn Fn()> = tests
            .iter()
            .filter_map(|test| match test {
                UzuTest::Bench(test) => Some(*test),
                UzuTest::Test(_) => None,
            })
            .collect::<Vec<_>>();
        criterion::runner(bench_tests.as_slice());
    } else {
        let default_tests: Vec<&TestDescAndFn> = tests
            .iter()
            .filter_map(|test| match test {
                UzuTest::Bench(_) => None,
                UzuTest::Test(test) => Some(*test),
            })
            .collect::<Vec<_>>();
        test::test_main_static(&default_tests)
    }
}
