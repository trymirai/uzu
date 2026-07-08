//! IOReport encodes a channel's unit as a `u64` returned by
//! `IOReportChannelGetUnit`. The top byte is the quantity kind and, for decimal
//! quantities, byte 4 holds an SI scale exponent biased by 127:
//!
//! ```text
//!   bits 56..64 : quantity kind        (3 = Energy / Joules)
//!   bits 32..40 : 127 + SI exponent    (124 -> 1e-3 "mJ", 118 -> 1e-9 "nJ")
//! ```
//!
//! Cross-checked three ways: libIOReport's `copyUnitLabel` switches on exactly
//! these codes (`0x7C<<32` -> "m", `0x76<<32` -> "n", `0x85<<32` -> "M", ...),
//! its `IOReportScaleValue` derives factors from the same field, and the
//! open-source `gpucap` decoder uses the identical layout in production.

const QUANTITY_ENERGY: u64 = 3;
const SI_EXPONENT_BIAS: i32 = 127;

pub(crate) fn energy_joules(
    value: i64,
    unit: u64,
) -> f64 {
    if (unit >> 56) & 0xff != QUANTITY_ENERGY {
        return 0.0;
    }
    let exponent = (((unit >> 32) & 0xff) as i32) - SI_EXPONENT_BIAS;
    value as f64 * 10f64.powi(exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an energy unit code the way libIOReport encodes it.
    const fn energy_unit(si_exponent_offset: u64) -> u64 {
        (QUANTITY_ENERGY << 56) | (si_exponent_offset << 32)
    }

    #[test]
    fn decodes_the_reversed_dylib_constants() {
        // The exact codes libIOReport's copyUnitLabel matches, and their scale.
        assert_eq!(energy_joules(1, energy_unit(0x73)), 1e-12); // p
        assert_eq!(energy_joules(1, energy_unit(0x76)), 1e-9); // n
        assert_eq!(energy_joules(1, energy_unit(0x79)), 1e-6); // u
        assert_eq!(energy_joules(1, energy_unit(0x7c)), 1e-3); // m
        assert_eq!(energy_joules(1, energy_unit(0x7f)), 1e0); // (none)
        assert_eq!(energy_joules(1, energy_unit(0x82)), 1e3); // k
        assert_eq!(energy_joules(1, energy_unit(0x85)), 1e6); // M
    }

    #[test]
    fn scales_the_raw_value() {
        // 5000 mJ -> 5 J, 2_000_000 nJ -> 0.002 J.
        assert_eq!(energy_joules(5000, energy_unit(0x7c)), 5.0);
        assert_eq!(energy_joules(2_000_000, energy_unit(0x76)), 0.002);
    }

    #[test]
    fn rejects_non_energy_quantities() {
        let ticks = (7u64 << 56) | (0x7f << 32);
        assert_eq!(energy_joules(1234, ticks), 0.0);
        assert_eq!(energy_joules(1234, 0), 0.0);
        // The error sentinel IOReportChannelGetUnit returns on failure.
        assert_eq!(energy_joules(1234, 0x8000000000000000), 0.0);
    }
}
