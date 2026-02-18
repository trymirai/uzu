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
        let config_path = model_path.join("config.json");
        if !config_path.exists() {
            return Err(Error::ModelFolderNotFound);
        }

        let config_file = std::fs::File::open(&config_path).map_err(|_| Error::UnableToLoadConfig)?;
        let model_metadata: ModelMetadata =
            serde_json::from_reader(std::io::BufReader::new(config_file)).map_err(|_| Error::UnableToLoadConfig)?;

        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(Error::UnableToLoadTokenizer);
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|_| Error::UnableToLoadTokenizer)?;

        let classifier_model_config = model_metadata.model_config.as_classifier().ok_or(Error::UnableToLoadConfig)?;
        let input_processor =
            Box::new(InputProcessorDefault::new(classifier_model_config.message_processor_config.clone()))
                as Box<dyn InputProcessor>;
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
        let preprocessing_start = std::time::Instant::now();
        let text = self.input_processor.process(&input, false, false)?;
        let tokens: Vec<u64> = self
            .tokenizer
            .encode(text.as_str(), false)
            .map_err(|_| Error::UnableToEncodeText)?
            .get_ids()
            .iter()
            .map(|&id| id as u64)
            .collect();

        let token_positions: Vec<usize> = (0..tokens.len()).collect();
        let preprocessing_duration = preprocessing_start.elapsed().as_secs_f64();

        let mut output = self.classifier.classify_tokens(tokens, token_positions)?;
        output.stats.preprocessing_duration = preprocessing_duration;

        Ok(output)
    }
}
