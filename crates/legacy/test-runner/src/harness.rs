use crate::util::enable_benchmark_gpu_capture_if_requested;

pub enum UzuTest {
    Bench(&'static dyn Fn()),
    Test(&'static test::TestDescAndFn),
}

#[cfg(target_os = "ios")]
pub fn ios_set_current_dir() {
    use objc2_foundation::{NSSearchPathDirectory, NSSearchPathDomainMask, NSSearchPathForDirectoriesInDomains};
    let paths = NSSearchPathForDirectoriesInDomains(
        NSSearchPathDirectory(9),  // NSDocumentDirectory
        NSSearchPathDomainMask(1), // NSUserDomainMask
        true,
    );
    if let Some(docs) = paths.firstObject() {
        let _ = std::env::set_current_dir(docs.to_string());
    }
}

pub fn uzu_harness(tests: &[&UzuTest]) {
    let args = std::env::args().collect::<Vec<String>>();
    let benchmarks = args.contains(&"--bench".to_string());
    if benchmarks {
        #[cfg(target_os = "ios")]
        uzu_engine::tests::path::ios_set_current_dir();
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
