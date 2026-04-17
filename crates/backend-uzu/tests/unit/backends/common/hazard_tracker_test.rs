use super::*;

#[test]
fn test_hazard_tracker_simple_compute_sequence() {
    let mut ht = HazardTracker::new();

    // RAR is fine (many readers don't interfere with each other)
    for _ in 0..123 {
        assert_eq!(
            ht.access(&[Access {
                range: 0..16,
                flags: AccessFlags::compute_read()
            }]),
            None
        );
    }

    // WAR causes barrier after reads before writes (to ensure that the reads are done before writes can rewrite the data)
    assert_eq!(
        ht.access(&[Access {
            range: 0..16,
            flags: AccessFlags::compute_write()
        }]),
        Some((AccessFlags::compute_read(), AccessFlags::compute_write()))
    );

    // WAW causes barrier after writes before writes (to ensure the last write wins)
    for _ in 0..123 {
        assert_eq!(
            ht.access(&[Access {
                range: 0..16,
                flags: AccessFlags::compute_write()
            }]),
            Some((AccessFlags::compute_write(), AccessFlags::compute_write()))
        );
    }

    // RAW causes barrier after writes before reads (to ensure that writes wrote the data before reads read it)
    assert_eq!(
        ht.access(&[Access {
            range: 0..16,
            flags: AccessFlags::compute_read()
        }]),
        Some((AccessFlags::compute_write(), AccessFlags::compute_read()))
    );
}
