use std::{path::PathBuf, pin::Pin, sync::Arc};

use shoji::{
    traits::backend::{Error as BackendError, decision::DecisionInstance},
    types::session::classification::TokenCodecConfig,
};
use tokenizers::Tokenizer;
use uzu_engine::{
    backends::Backend,
    engine::{Engine, decision_model::DecisionModel},
};

pub struct UzuDecisionBackendInstance<B: Backend> {
    model: Arc<DecisionModel<B>>,
}

impl<B: Backend> UzuDecisionBackendInstance<B> {
    pub async fn new(model_path: String) -> Result<Self, BackendError> {
        let engine = Engine::<B>::new().await.map_err(|err| err.to_string())?;
        let model_path = PathBuf::from(model_path);
        let model = engine.load_decision_model(&model_path).await.map_err(|err| err.to_string())?;
        Ok(Self {
            model: Arc::new(model),
        })
    }
}

impl<B: Backend> DecisionInstance for UzuDecisionBackendInstance<B> {
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.model.tokenizer().clone()
    }

    fn token_codec_config(&self) -> TokenCodecConfig {
        self.model.token_codec_config()
    }

    fn codes(&self) -> Vec<String> {
        self.model.decision_config().codes.to_vec()
    }

    fn probabilities<'a>(
        &'a self,
        input: &'a [u64],
        option_count: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, BackendError>> + Send + 'a>> {
        Box::pin(async move {
            self.model.probabilities(input, option_count).await.map_err(|err| BackendError::from(err.to_string()))
        })
    }
}
