use crate::session::types::PowerStats;

pub(crate) struct PowerRecorder {
    #[cfg(target_vendor = "apple")]
    collector: keisoku::Collector,
    #[cfg(target_vendor = "apple")]
    total_window: Option<keisoku::EnergyWindow>,
    #[cfg(target_vendor = "apple")]
    prefill_reading: Option<keisoku::EnergyReading>,
    #[cfg(target_vendor = "apple")]
    decode_reading: Option<keisoku::EnergyReading>,
}

impl PowerRecorder {
    pub(crate) fn start() -> Self {
        #[cfg(target_vendor = "apple")]
        {
            let collector = keisoku::Collector::new();
            let total_window = collector.start_energy_window();
            Self {
                collector,
                total_window,
                prefill_reading: None,
                decode_reading: None,
            }
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            Self {}
        }
    }

    pub(crate) fn start_phase(&self) -> PhaseWindow {
        #[cfg(target_vendor = "apple")]
        {
            PhaseWindow {
                window: self.collector.start_energy_window(),
            }
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            PhaseWindow {}
        }
    }

    pub(crate) fn finish_prefill(
        &mut self,
        window: PhaseWindow,
    ) {
        #[cfg(target_vendor = "apple")]
        {
            self.prefill_reading = window.window.and_then(|window| self.collector.end_energy_window(window));
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = window;
        }
    }

    pub(crate) fn finish_decode(
        &mut self,
        window: PhaseWindow,
    ) {
        #[cfg(target_vendor = "apple")]
        {
            self.decode_reading = window.window.and_then(|window| self.collector.end_energy_window(window));
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = window;
        }
    }

    pub(crate) fn stop(mut self) -> Option<PowerStats> {
        #[cfg(target_vendor = "apple")]
        {
            let total_reading = self.total_window.take().and_then(|window| self.collector.end_energy_window(window))?;
            PowerStats::from_energy_readings(total_reading, self.prefill_reading, self.decode_reading)
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            None
        }
    }
}

pub(crate) struct PhaseWindow {
    #[cfg(target_vendor = "apple")]
    window: Option<keisoku::EnergyWindow>,
}
