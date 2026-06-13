#![cfg(test)]
#![allow(dead_code)]

pub mod assert;
pub mod env_vars;
pub mod helpers;
pub mod metrics;
pub mod perf;
pub mod util;

pub extern crate test;

use crate::tests::util::enable_benchmark_gpu_capture_if_requested;

pub enum UzuTest {
    Bench(&'static dyn Fn()),
    Test(&'static test::TestDescAndFn),
}

pub fn uzu_harness(tests: &[&UzuTest]) {
    let args = std::env::args().collect::<Vec<String>>();
    let benchmarks = args.contains(&"--bench".to_string());
    if benchmarks {
        #[cfg(target_os = "ios")]
        crate::common::path::ios_set_current_dir();
        enable_benchmark_gpu_capture_if_requested();
        let bench_tests: Vec<&dyn Fn()> = tests
            .iter()
            .filter_map(|test| match test {
                UzuTest::Bench(test) => Some(*test),
                UzuTest::Test(_) => None,
            })
            .collect::<Vec<_>>();
        criterion::runner(bench_tests.as_slice());
    } else {
        let default_tests: Vec<&test::TestDescAndFn> = tests
            .iter()
            .filter_map(|test| match test {
                UzuTest::Bench(_) => None,
                UzuTest::Test(test) => Some(*test),
            })
            .collect::<Vec<_>>();
        test::test_main_static(&default_tests)
    }
}
