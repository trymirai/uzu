use proc_macros::uzu_test;

use super::*;

#[uzu_test]
fn test_hazard_tracker_simple_compute_sequence() {
    let mut ht = HazardTracker::new();

    // RAR is fine (many readers don't interfere with each other)
    for _ in 0..123 {
        assert_eq!(
            ht.access(&[Access {
                range: 0..16,
                flags: AccessFlags::compute_read(),
                resource: None,
            }])
            .map(|barrier| (barrier.after, barrier.before)),
            None
        );
    }

    // WAR causes barrier after reads before writes (to ensure that the reads are done before writes can rewrite the data)
    assert_eq!(
        ht.access(&[Access {
            range: 0..16,
            flags: AccessFlags::compute_write(),
            resource: None,
        }])
        .map(|barrier| (barrier.after, barrier.before)),
        Some((AccessFlags::compute_read(), AccessFlags::compute_write()))
    );

    // WAW causes barrier after writes before writes (to ensure the last write wins)
    for _ in 0..123 {
        assert_eq!(
            ht.access(&[Access {
                range: 0..16,
                flags: AccessFlags::compute_write(),
                resource: None,
            }])
            .map(|barrier| (barrier.after, barrier.before)),
            Some((AccessFlags::compute_write(), AccessFlags::compute_write()))
        );
    }

    // RAW causes barrier after writes before reads (to ensure that writes wrote the data before reads read it)
    assert_eq!(
        ht.access(&[Access {
            range: 0..16,
            flags: AccessFlags::compute_read(),
            resource: None,
        }])
        .map(|barrier| (barrier.after, barrier.before)),
        Some((AccessFlags::compute_write(), AccessFlags::compute_read()))
    );
}
