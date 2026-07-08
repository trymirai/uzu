use super::{Channel, ChannelFold, FrequencyTables, Rail, RawChannel, energy_joules};

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
        let joules = energy_joules(raw.integer_value, raw.unit);
        match rail {
            Rail::Cpu => self.cpu += joules,
            Rail::Gpu => self.gpu += joules,
            Rail::Ane => self.ane += joules,
            Rail::Ram => self.ram += joules,
        }
    }
}
