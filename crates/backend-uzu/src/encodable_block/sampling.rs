use std::{
    cell::RefCell,
    collections::{HashMap, hash_map::Entry},
    mem::size_of,
};

use crate::{
    backends::common::{Allocation, AllocationType, Backend, Context, Encoder, Kernels, kernel::UnifiedSamplingKernel},
    data_type::DataType,
    session::parameter::{SamplingMethod, SamplingProcessingOrder},
};

#[derive(PartialEq, Eq, Hash)]
struct UnifiedSamplingKey {
    has_seeds: bool,
    has_bitmask: bool,
    has_temperature: bool,
    temperature_after_filters: bool,
    has_top_k: bool,
    has_top_p: bool,
    has_min_p: bool,
}

pub struct Sampling<B: Backend> {
    vocab_size: usize,
    data_type: DataType,
    unified_kernels: RefCell<HashMap<UnifiedSamplingKey, <B::Kernels as Kernels>::UnifiedSamplingKernel>>,
}

impl<B: Backend> Sampling<B> {
    pub fn new(
        data_type: DataType,
        vocab_size: usize,
    ) -> Self {
        Self {
            vocab_size,
            data_type,
            unified_kernels: RefCell::new(HashMap::new()),
        }
    }

    pub fn encode(
        &self,
        logits: &Allocation<B>,
        seeds: Option<&Allocation<B>>,
        bitmask: Option<&Allocation<B>>,
        sampling_method: SamplingMethod,
        batch_size: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        // TODO: reject seeds with greedy
        let (seeds, temperature, temperature_after_filters, top_k, top_p, min_p) = match sampling_method {
            SamplingMethod::Greedy => (None, None, false, None, None, None),
            SamplingMethod::Stochastic {
                temperature,
                top_k,
                top_p,
                min_p,
                processing_order,
            } => (
                Some(seeds.unwrap()),
                temperature,
                match processing_order {
                    SamplingProcessingOrder::TemperatureThenFilters => false,
                    SamplingProcessingOrder::FiltersThenTemperature => true,
                },
                top_k,
                top_p,
                min_p,
            ),
        };

        let key = UnifiedSamplingKey {
            has_seeds: seeds.is_some(),
            has_bitmask: bitmask.is_some(),
            has_temperature: temperature.is_some(),
            temperature_after_filters,
            has_top_k: top_k.is_some(),
            has_top_p: top_p.is_some(),
            has_min_p: min_p.is_some(),
        };
        let mut unified_kernels = self.unified_kernels.borrow_mut();
        let entry = unified_kernels.entry(key);
        let kernel = match entry {
            Entry::Occupied(occupied) => occupied.into_mut(),
            Entry::Vacant(vacant) => {
                let key = vacant.key();

                let kernel = <B::Kernels as Kernels>::UnifiedSamplingKernel::new(
                    encoder.context(),
                    self.data_type,
                    key.has_seeds,
                    key.has_bitmask,
                    key.has_temperature,
                    key.temperature_after_filters,
                    key.has_top_k,
                    key.has_top_p,
                    key.has_min_p,
                )?;

                vacant.insert(kernel)
            },
        };

        let mut output = encoder.context().create_allocation(batch_size * size_of::<u32>(), AllocationType::Global)?;

        kernel.encode(
            logits,
            &mut output,
            seeds,
            bitmask,
            temperature,
            top_k,
            top_p,
            min_p,
            self.vocab_size as u32,
            batch_size as u32,
            encoder,
        );

        Ok(output)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/encodable_block/sampling_test.rs"]
mod tests;
