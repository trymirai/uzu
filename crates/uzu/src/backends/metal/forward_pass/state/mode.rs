use std::{cell::RefCell, rc::Rc};

use super::super::cache_layers::CacheLayers;
#[cfg(feature = "tracing")]
use super::super::traces::ActivationTrace;
use crate::{backends::metal::MetalArray, session::parameter::SamplingMethod};

type ArrayCell = RefCell<MetalArray>;

pub enum ForwardPassMode {
    LanguageModelGenerator(LanguageModelGeneratorModeState),
    Classifier(ClassifierModeState),
}

pub struct LanguageModelGeneratorModeState {
    pub cache_layers: Rc<RefCell<CacheLayers>>,
    pub token_seeds: ArrayCell,
    pub logits: ArrayCell,
    pub sampling_output: Option<ArrayCell>,
    pub sampling_method: Option<SamplingMethod>,
    #[cfg(feature = "tracing")]
    pub traces: Rc<RefCell<ActivationTrace>>,
    pub active_suffix_length: usize,
    pub sampling_start: usize,
    pub sampling_length: usize,
    pub is_prefilling: bool,
}

pub struct ClassifierModeState {
    pub pooling: ArrayCell,
    pub dense: ArrayCell,
    pub norm: ArrayCell,
    pub classifier_logits: ArrayCell,
    #[cfg(feature = "tracing")]
    pub traces: Rc<RefCell<ActivationTrace>>,
}
