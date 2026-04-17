#![feature(custom_test_frameworks)]
#![test_runner(crate::bench_runner)]

fn bench_runner(benches: &[&dyn Fn()]) {
    #[cfg(target_os = "ios")]
    {
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
    criterion::runner(benches);
}

use dsl::{__internal_uzu_bench as uzu_bench, __internal_uzu_ignored as uzu_test};

include!("mod.rs");
