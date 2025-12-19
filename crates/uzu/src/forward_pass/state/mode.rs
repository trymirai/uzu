use std::{cell::RefCell, rc::Rc};

use super::super::cache_layers::CacheLayers;
#[cfg(feature = "tracing")]
use super::super::traces::ActivationTrace;
use crate::{DeviceContext, session::parameter::SamplingMethod};

type ArrayCell<C> = RefCell<<C as DeviceContext>::DeviceArray>;

pub enum ForwardPassMode<C: DeviceContext> {
    LanguageModelGenerator(LanguageModelGeneratorModeState<C>),
    Classifier(ClassifierModeState<C>),
}

pub struct LanguageModelGeneratorModeState<C: DeviceContext> {
    pub cache_layers: Rc<RefCell<CacheLayers<C>>>,
    pub token_seeds: ArrayCell<C>,
    pub logits: ArrayCell<C>,
    pub sampling_output: Option<ArrayCell<C>>,
    pub sampling_method: Option<SamplingMethod>,
    #[cfg(feature = "tracing")]
    pub traces: Rc<RefCell<ActivationTrace<C>>>,
    pub active_suffix_length: usize,
    pub is_prefilling: bool,
}

pub struct ClassifierModeState<C: DeviceContext> {
    pub pooling: ArrayCell<C>,
    pub dense: ArrayCell<C>,
    pub norm: ArrayCell<C>,
    pub classifier_logits: ArrayCell<C>,
    #[cfg(feature = "tracing")]
    pub traces: Rc<RefCell<ActivationTrace<C>>>,
}
