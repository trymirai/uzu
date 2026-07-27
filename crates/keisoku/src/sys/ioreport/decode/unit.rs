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
