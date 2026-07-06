use std::time::Instant;

use anyhow::{Result, anyhow};
use backend_uzu::{
    backends::metal::Metal,
    engine::language_model::{
        LanguageModel,
        stream::{LanguageModelStreamOptions, SamplingMethod},
    },
};
use keisoku::{EnergyMeter, EnergyReading};

pub struct Measurement {
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_tokens: usize,
    pub reading: Option<EnergyReading>,
}

pub fn run(
    model: &LanguageModel<Metal>,
    meter: &EnergyMeter,
    prefill: usize,
    generate: usize,
) -> Result<Measurement> {
    let input: Vec<u64> = (0..prefill).map(|index| (index % 7) as u64).collect();
    let mut state = model
        .create_empty_state(Some(prefill + generate + 1))
        .map_err(|error| anyhow!("create_empty_state: {error}"))?;
    let options = LanguageModelStreamOptions {
        sampling_method: SamplingMethod::Greedy,
        grammar: None,
        speculator: None,
    };

    let window = meter.start();
    let start = Instant::now();

    let mut stream = model.stream(&input, &mut state, options).map_err(|error| anyhow!("stream: {error}"))?;

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

    let reading = meter.stop(window);

    Ok(Measurement {
        prefill_ms: (after_prefill - start).as_secs_f64() * 1000.0,
        decode_ms: (end - after_prefill).as_secs_f64() * 1000.0,
        decode_tokens,
        reading,
    })
}
