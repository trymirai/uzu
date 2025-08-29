pub mod argmax;
pub mod common;
pub use common::{CPUArray, CPUContext};

use crate::backends::Backend;

pub struct CPUBackend {}
pub struct CPUState {}

impl Backend for CPUBackend {
    type Array = CPUArray;

    type Context = CPUContext;

    type State = CPUState;

    fn new(
        _context: std::rc::Rc<Self::Context>,
        _generator_config: &crate::generator::config::GeneratorConfig,
        _decoder_config: &crate::config::DecoderConfig,
        _weights: &crate::parameters::ParameterTree<Self>,
    ) -> Result<Self, crate::generator::error::GeneratorError> {
        todo!()
    }

    fn context(&self) -> &Self::Context {
        todo!()
    }

    fn run(
        &mut self,
        _tokens: &[u64],
        _token_positions: &[usize],
        _expected_amount_of_new_tokens: usize,
        _sampling_config: Option<super::SamplingConfig>,
        _warmup: bool,
    ) -> super::RunResult<Self::Array> {
        todo!()
    }

    fn accept_tokens(
        &mut self,
        _indices: &[usize],
    ) {
        todo!()
    }

    fn clone_state(&self) -> Self::State {
        todo!()
    }

    fn restore_state(
        &mut self,
        _state: &Self::State,
    ) {
        todo!()
    }

    fn reset_state(&mut self) {
        todo!()
    }

    fn prefix_length(&self) -> usize {
        todo!()
    }
}
