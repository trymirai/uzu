use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::Backend,
    engine::{
        Engine,
        language_model::{LanguageModel, state::LanguageModelState, stream::SamplingMethod},
    },
    tests::{helpers::for_each_non_cpu_backend, path::get_test_model_path},
};

fn generate<B: Backend>(
    model: &LanguageModel<B>,
    state: &mut LanguageModelState<B>,
    input: &[u64],
    snapshot_position: Option<usize>,
    count: usize,
) -> Vec<u64> {
    let mut options = model.default_stream_options();
    options.sampling_method = SamplingMethod::Greedy;
    options.snapshot_position = snapshot_position;
    model.stream(input, state, options).unwrap().take(count).map(Result::unwrap).collect()
}

#[uzu_test]
fn rewind_to_snapshot_matches_fresh_prefill() {
    for_each_non_cpu_backend!(|B| {
        let engine = Engine::<B>::new().unwrap();
        let model = engine.load_language_model(&get_test_model_path()).unwrap();
        if !model.snapshot_supported() {
            return;
        }
        let encode = |text: &str| -> Vec<u64> {
            model.tokenizer().encode(text, false).unwrap().get_ids().iter().map(|&id| u64::from(id)).collect()
        };
        let prompt = encode(&"The quick brown fox jumps over the lazy dog. ".repeat(40));
        let other = encode(" Now write a haiku about it.");
        let snapshot = prompt.len() / 2;

        // Past the snapshot (the rest of the prompt and a reply), then back to it.
        let mut rewound = model.create_empty_state(None, 0).unwrap();
        generate(&model, &mut rewound, &prompt, Some(snapshot), 16);
        let context = [&prompt[..snapshot], &other[..]].concat();
        assert_eq!(model.rewind(&mut rewound, &context).unwrap(), Some(snapshot));
        let after_rewind = generate(&model, &mut rewound, &other, None, 16);

        // The shared part prefilled in the same batches, and nothing else.
        let mut fresh = model.create_empty_state(None, 0).unwrap();
        generate(&model, &mut fresh, &prompt[..snapshot], None, 0);
        assert_eq!(fresh.tokens(), &prompt[..snapshot]);
        let after_prefill = generate(&model, &mut fresh, &other, None, 16);

        assert_eq!(after_rewind, after_prefill);

        // A context that extends the state continues it; one that shares less than the snapshot cannot be reused.
        let length = rewound.tokens().len();
        let extended = [rewound.tokens(), &other[..]].concat();
        assert_eq!(model.rewind(&mut rewound, &extended).unwrap(), Some(length));
        assert_eq!(model.rewind(&mut rewound, &other).unwrap(), None);
    });
}
