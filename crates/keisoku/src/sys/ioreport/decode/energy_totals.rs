use obfstr::obfstr;

#[derive(Default, Clone, Copy)]
pub struct EnergyTotals {
    pub(crate) cpu: f64,
    pub(crate) gpu: f64,
    pub(crate) ane: f64,
    pub(crate) ram: f64,
}

impl EnergyTotals {
    pub(crate) fn accumulate(
        &mut self,
        name: &str,
        value: i64,
        unit: &str,
    ) {
        let joules = joules(value, unit);
        if name == obfstr!("GPU Energy") {
            self.gpu += joules;
        } else if name.ends_with(obfstr!("CPU Energy")) {
            self.cpu += joules;
        } else if name.starts_with(obfstr!("ANE")) {
            self.ane += joules;
        } else if name.starts_with(obfstr!("DRAM"))
            || name.starts_with(obfstr!("DCS"))
            || name.starts_with(obfstr!("AMCC"))
        {
            self.ram += joules;
        }
    }

    pub(crate) fn total(&self) -> f64 {
        self.cpu + self.gpu + self.ane + self.ram
    }
}

fn joules(
    energy: i64,
    unit: &str,
) -> f64 {
    let energy = energy as f64;
    let Some(prefix) = unit.trim().strip_suffix('J') else {
        return 0.0;
    };
    let scale = match prefix {
        "k" => 1e3,
        "" => 1.0,
        "m" => 1e-3,
        "u" | "µ" => 1e-6,
        "n" => 1e-9,
        "p" => 1e-12,
        _ => return 0.0,
    };
    energy * scale
}
