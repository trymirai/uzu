use std::sync::Arc;

pub type EngineCallbackType = Box<dyn Fn() + Send + Sync>;

#[bindings::export(Class)]
#[derive(Clone)]
pub struct EngineCallback {
    callback: Arc<EngineCallbackType>,
}

impl EngineCallback {
    pub fn new(callback: EngineCallbackType) -> Self {
        Self {
            callback: Arc::new(callback),
        }
    }

    pub fn on_event(&self) {
        (self.callback)();
    }
}

#[bindings::export(Implementation)]
impl EngineCallback {
    #[bindings::export(Method(FactoryWithCallback))]
    pub fn create(callback: Box<dyn Fn() + Send + Sync>) -> Self {
        Self::new(callback)
    }
}
