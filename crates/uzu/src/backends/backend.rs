use std::rc::Rc;

use half::{bf16, f16};
use num_traits::Zero;

use crate::{
    Array, DataType,
    backends::{self, Context, SamplingConfig},
    config::DecoderConfig,
    generator::{config::GeneratorConfig, error::GeneratorError},
    parameters::ParameterTree,
};

pub struct RunResult<A: Array> {
    pub sampling_output: Option<A>,
    pub duration: f64,
}

pub trait Backend
where
    Self: Sized,
{
    type Array: self::Array;
    type Context: backends::Context<Self>;

    type State;

    fn new(
        context: Rc<Self::Context>,
        generator_config: &GeneratorConfig,
        decoder_config: &DecoderConfig,
        weights: &ParameterTree<Self>,
    ) -> Result<Self, GeneratorError>;

    fn context(&self) -> &Self::Context;

    fn run(
        &mut self,
        tokens: &[u64],
        token_positions: &[usize],
        expected_amount_of_new_tokens: usize,
        sampling_config: Option<SamplingConfig>,
        warmup: bool,
    ) -> RunResult<Self::Array>;

    fn accept_tokens(
        &mut self,
        indices: &[usize],
    );

    fn clone_state(&self) -> Self::State;

    fn restore_state(
        &mut self,
        state: &Self::State,
    );

    fn reset_state(&mut self);

    fn prefix_length(&self) -> usize;

    /// Allocate a new array for an attention bias (also sometimes referred to as "attention mask").
    /// The result is -inf at indices where `should_be_neg_inf` is true, and zero otherwise.
    fn attention_bias<F>(
        &self,
        suffix_length: usize,
        prefix_length: usize,
        data_type: DataType,
        mut should_be_neg_inf: F,
    ) -> Self::Array
    where
        F: FnMut(usize, usize) -> bool,
    {
        let shape = [suffix_length, suffix_length + prefix_length];
        match data_type {
            DataType::F16 => {
                self.context().array_from_shape_fn(&shape, |[row, col]| {
                    if should_be_neg_inf(*row, *col) {
                        f16::NEG_INFINITY
                    } else {
                        f16::zero()
                    }
                })
            },
            DataType::BF16 => {
                self.context().array_from_shape_fn(&shape, |[row, col]| {
                    if should_be_neg_inf(*row, *col) {
                        bf16::NEG_INFINITY
                    } else {
                        bf16::zero()
                    }
                })
            },
            DataType::F32 => {
                self.context().array_from_shape_fn(&shape, |[row, col]| {
                    if should_be_neg_inf(*row, *col) {
                        f32::NEG_INFINITY
                    } else {
                        f32::zero()
                    }
                })
            },
            DataType::F64 => {
                self.context().array_from_shape_fn(&shape, |[row, col]| {
                    if should_be_neg_inf(*row, *col) {
                        f64::NEG_INFINITY
                    } else {
                        f64::zero()
                    }
                })
            },
            _ => {
                panic!(
                    "Attention bias can only be of a floating-point type, but requested {:?}",
                    data_type
                );
            },
        }
    }
}
