use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{backends::cpu::Cpu, engine::Engine, tests::path::get_test_model_path};

/// The CPU verifies no speculation trees, so a TEST_MODEL that ships a speculator, such as Qwen3.5 4B Mirai-M or
/// Muse-Glimmer 30B Mirai-M, loads without it and reports it unavailable.
#[uzu_test]
#[ignore] // Needs a TEST_MODEL checkpoint with a speculator.
fn cpu_loads_a_model_without_its_speculator() {
    let model_path = get_test_model_path();
    assert!(model_path.join("speculator").is_dir(), "{} ships no speculator", model_path.display());
    let engine = block_on(Engine::<Cpu>::new()).expect("engine");
    let model = block_on(engine.load_language_model(&model_path)).expect("language model");
    assert!(model.speculator.is_none());
    assert!(model.speculator_unavailable());
}
