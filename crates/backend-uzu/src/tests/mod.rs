#![allow(dead_code)]

#[cfg(test)]
pub extern crate test;

pub enum UzuTest {
    Bench(&'static dyn Fn()),
    Test(&'static test::TestDescAndFn),
}

#[cfg(test)]
pub fn uzu_harness(tests: &[&UzuTest]) {
    let args = std::env::args().collect::<Vec<String>>();
    let benchmarks = args.contains(&"--bench".to_string());
    if benchmarks {
        #[cfg(target_os = "ios")]
        crate::common::path::ios_set_current_dir();
        crate::common::enable_benchmark_gpu_capture_if_requested();
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
