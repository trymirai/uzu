use std::time::Instant;

use anyhow::{Result, anyhow};
use keisoku::{Ane, Cpu, Device, DramBytes, DramHistogram, DramRead, DramWrite, EnergyRail, Gpu, Ram, Select};
use uzu_engine::{
    backends::metal::Metal,
    engine::language_model::{LanguageModel, stream::SamplingMethod},
};

use super::reading::Reading;

const SAMPLING_SEED: u64 = 0;

pub struct Measurement {
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_tokens: usize,
    pub reading: Option<Reading>,
}

impl Measurement {
    /// Runs one prefill/generate pass, timing the two phases and metering the whole window.
    pub fn capture(
        model: &LanguageModel<Metal>,
        input: &mut Vec<u64>,
        prefill: usize,
        generate: usize,
    ) -> Result<Self> {
        input.clear();
        input.extend((0..prefill).map(|index| (index % 7) as u64));

        let mut state = model
            .create_empty_state(model.recommended_context_length(), SAMPLING_SEED)
            .map_err(|error| anyhow!("create_empty_state: {error}"))?;
        let mut options = model.default_stream_options();
        options.sampling_method = SamplingMethod::Greedy;

        let mut probe = Device::interval_measurement::<
            Select![
                EnergyRail<Cpu>,
                EnergyRail<Gpu>,
                EnergyRail<Ane>,
                EnergyRail<Ram>,
                DramBytes<DramRead>,
                DramBytes<DramWrite>,
                DramHistogram<DramRead>,
                DramHistogram<DramWrite>,
            ],
        >();
        probe.start();
        let start = Instant::now();

        let mut stream = model.stream(input, &mut state, options).map_err(|error| anyhow!("stream: {error}"))?;

        match stream.next() {
            Some(Ok(_)) => {},
            Some(Err(error)) => return Err(anyhow!("prefill: {error}")),
            None => {},
        }
        let after_prefill = Instant::now();

        let mut decode_tokens = 0usize;
        for _ in 1..generate {
            match stream.next() {
                Some(Ok(_)) => decode_tokens += 1,
                Some(Err(error)) => return Err(anyhow!("decode: {error}")),
                None => break,
            }
        }
        let end = Instant::now();
        drop(stream);

        let reading = probe.stop().map(|sample| Reading {
            cpu_j: f64::from(sample.get::<EnergyRail<Cpu>>().value()),
            gpu_j: f64::from(sample.get::<EnergyRail<Gpu>>().value()),
            ane_j: f64::from(sample.get::<EnergyRail<Ane>>().value()),
            ram_j: f64::from(sample.get::<EnergyRail<Ram>>().value()),
            dram_read_bytes: sample.get::<DramBytes<DramRead>>().value(),
            dram_write_bytes: sample.get::<DramBytes<DramWrite>>().value(),
            dram_read_gbps: f64::from(sample.get::<DramHistogram<DramRead>>().value()),
            dram_write_gbps: f64::from(sample.get::<DramHistogram<DramWrite>>().value()),
        });

        Ok(Self {
            prefill_ms: (after_prefill - start).as_secs_f64() * 1000.0,
            decode_ms: (end - after_prefill).as_secs_f64() * 1000.0,
            decode_tokens,
            reading,
        })
    }
}
