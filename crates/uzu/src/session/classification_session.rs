use std::path::PathBuf;

use tokenizers::Tokenizer;

use crate::{
    classifier::{ClassificationOutput, Classifier},
    config::ModelMetadata,
    session::{
        helpers::{InputProcessor, InputProcessorDefault},
        types::{Error, Input},
    },
};

pub struct ClassificationSession {
    #[allow(dead_code)]
    model_path: PathBuf,
    #[allow(dead_code)]
    model_metadata: ModelMetadata,
    tokenizer: Tokenizer,
    classifier: Classifier,
    input_processor: Box<dyn InputProcessor>,
}

impl ClassificationSession {
    pub fn new(model_path: PathBuf) -> Result<Self, Error> {
        // Load model metadata
        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(Error::ModelFolderNotFound);
        }

        let config_file = std::fs::File::open(&config_path)
            .map_err(|_| Error::UnableToLoadConfig)?;
        let model_metadata: ModelMetadata =
            serde_json::from_reader(std::io::BufReader::new(config_file))
                .map_err(|_| Error::UnableToLoadConfig)?;

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::UnableToLoadTokenizer);
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|_| Error::UnableToLoadTokenizer)?;
        // Keep tokenizer padding enabled (pads to 2048) to match Lalamo's behavior.
        // Mean pooling in Lalamo averages over all tokens including padding.

        // Extract classifier model config
        let classifier_model_config = model_metadata
            .model_config
            .as_classifier()
            .ok_or(Error::UnableToLoadConfig)?;

        // Create input processor using message processor config
        let input_processor = Box::new(InputProcessorDefault::new(
            classifier_model_config.message_processor_config.clone(),
        )) as Box<dyn InputProcessor>;

        // Create classifier
        let classifier = Classifier::new(&model_path)?;

        Ok(Self {
            model_path,
            model_metadata,
            tokenizer,
            classifier,
            input_processor,
        })
    }

    pub fn classify(
        &mut self,
        input: Input,
    ) -> Result<ClassificationOutput, Error> {
        // Debug print removed
        let text = self.input_processor.process(&input, false)?;

        // Debug print removed
        let tokens: Vec<u64> = self
            .tokenizer
            .encode(text.as_str(), false)
            .map_err(|_| Error::UnableToEncodeText)?
            .get_ids()
            .iter()
            .map(|&id| id as u64)
            .collect();
        // Debug print removed

        let token_positions: Vec<usize> = (0..tokens.len()).collect();

        // Debug print removed
        self.classifier.classify_tokens(tokens, token_positions)
    }
}
