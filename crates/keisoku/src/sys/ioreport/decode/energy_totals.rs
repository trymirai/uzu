use super::{Channel, ChannelFold, FrequencyTables, Rail, RawChannel};

#[derive(Default, Clone, Copy)]
pub struct EnergyTotals {
    pub(crate) cpu: f64,
    pub(crate) gpu: f64,
    pub(crate) ane: f64,
    pub(crate) ram: f64,
}

impl ChannelFold for EnergyTotals {
    fn fold(
        &mut self,
        channel: Channel,
        raw: &RawChannel,
        _frequencies: Option<&FrequencyTables<'_>>,
    ) {
        let Channel::EnergyRail(rail) = channel else {
            return;
        };
        let joules = joules(raw.integer_value, &raw.unit);
        match rail {
            Rail::Cpu => self.cpu += joules,
            Rail::Gpu => self.gpu += joules,
            Rail::Ane => self.ane += joules,
            Rail::Ram => self.ram += joules,
        }
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
