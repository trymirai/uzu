const QUANTITY_ENERGY: u64 = 3;
const QUANTITY_SHIFT: u32 = 56;
const SI_SCALE_SHIFT: u32 = 32;
const SI_EXPONENT_BIAS: i32 = 127;

pub(crate) fn energy_joules(
    value: i64,
    unit: u64,
) -> f64 {
    if (unit >> QUANTITY_SHIFT) & 0xff != QUANTITY_ENERGY {
        return 0.0;
    }
    let exponent = (((unit >> SI_SCALE_SHIFT) & 0xff) as i32) - SI_EXPONENT_BIAS;
    value as f64 * 10f64.powi(exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn energy_unit(si_exponent_offset: u64) -> u64 {
        (QUANTITY_ENERGY << QUANTITY_SHIFT) | (si_exponent_offset << SI_SCALE_SHIFT)
    }

    #[test]
    fn decodes_the_reversed_dylib_constants() {
        assert_eq!(energy_joules(1, energy_unit(0x73)), 1e-12);
        assert_eq!(energy_joules(1, energy_unit(0x76)), 1e-9);
        assert_eq!(energy_joules(1, energy_unit(0x79)), 1e-6);
        assert_eq!(energy_joules(1, energy_unit(0x7c)), 1e-3);
        assert_eq!(energy_joules(1, energy_unit(0x7f)), 1e0);
        assert_eq!(energy_joules(1, energy_unit(0x82)), 1e3);
        assert_eq!(energy_joules(1, energy_unit(0x85)), 1e6);
    }

    #[test]
    fn scales_the_raw_value() {
        assert_eq!(energy_joules(5000, energy_unit(0x7c)), 5.0);
        assert_eq!(energy_joules(2_000_000, energy_unit(0x76)), 0.002);
    }

    #[test]
    fn rejects_non_energy_quantities() {
        let ticks = (7u64 << QUANTITY_SHIFT) | (0x7f << SI_SCALE_SHIFT);
        assert_eq!(energy_joules(1234, ticks), 0.0);
        assert_eq!(energy_joules(1234, 0), 0.0);
        assert_eq!(energy_joules(1234, 0x8000000000000000), 0.0);
    }
}
