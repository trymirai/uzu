pub(crate) extern crate test;

const METAL_CAPTURE_ENABLED: &str = "METAL_CAPTURE_ENABLED";
const UZU_CAPTURE_BENCH: &str = "UZU_CAPTURE_BENCH";

pub(crate) enum UzuTest {
    Bench(&'static dyn Fn()),
    Test(&'static test::TestDescAndFn),
}

#[cfg(target_os = "ios")]
fn ios_set_current_dir() {
    use objc2_foundation::{NSSearchPathDirectory, NSSearchPathDomainMask, NSSearchPathForDirectoriesInDomains};

    let paths = NSSearchPathForDirectoriesInDomains(NSSearchPathDirectory(9), NSSearchPathDomainMask(1), true);
    if let Some(docs) = paths.firstObject() {
        let _ = std::env::set_current_dir(docs.to_string());
    }
}

pub(crate) fn uzu_harness(tests: &[&UzuTest]) {
    let args = std::env::args().collect::<Vec<String>>();
    let benchmarks = args.contains(&"--bench".to_string());
    if benchmarks {
        #[cfg(target_os = "ios")]
        ios_set_current_dir();
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

fn enable_benchmark_gpu_capture_if_requested() {
    if enabled(UZU_CAPTURE_BENCH) {
        unsafe {
            std::env::set_var(METAL_CAPTURE_ENABLED, "1");
        }
    }
}

fn enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("yes") || v.eq_ignore_ascii_case("true"))
}
