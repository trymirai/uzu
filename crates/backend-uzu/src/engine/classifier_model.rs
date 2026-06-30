use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufReader},
    path::Path,
    rc::Rc,
};

use half::bf16;
use thiserror::Error;

use crate::{
    backends::common::{AllocationType, Backend, Context, Encoder},
    config::model::classifier_model::ClassifierModelConfig,
    data_type::DataType,
    encodable_block::classifier::{Classifier as ClassifierEncodable, ClassifierError as ClassifierEncodableError},
    engine::Engine,
    parameters::{HeaderLoadingError, ParameterLoader, ParameterLoaderError},
};

pub struct ClassifierModel<B: Backend> {
    context: Rc<B::Context>,
    classifier: ClassifierEncodable<B>,
    output_labels: Box<[String]>,
    data_type: DataType,
}

#[derive(Debug, Error)]
pub enum EngineLoadClassifierError<B: Backend> {
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("HeaderLoading error: {0}")]
    HeaderLoading(#[from] HeaderLoadingError),
    #[error("ParameterLoader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Classifier error: {0}")]
    Classifier(#[from] ClassifierEncodableError<B>),
}

impl<B: Backend> Engine<B> {
    pub fn load_classifier_model(
        &self,
        model_path: &Path,
    ) -> Result<ClassifierModel<B>, EngineLoadClassifierError<B>> {
        let context = self.context.clone();

        let config: ClassifierModelConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;

        let data_type = DataType::BF16;

        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weight_loader = ParameterLoader::new(&weights_file, context.as_ref())?;

        let classifier = ClassifierEncodable::new(
            context.as_ref(),
            &config.classifier_config,
            &weight_loader.tree().subtree("classifier")?,
            data_type,
        )?;

        let output_labels = if let Some(output_labels) = config.classifier_config.output_labels {
            assert!(output_labels.len() == config.classifier_config.num_labels);
            output_labels
        } else {
            (0..config.classifier_config.num_labels).map(|index| format!("class_{index}")).collect()
        };

        weight_loader.tree().assert_all_tensors_validated()?;

        Ok(ClassifierModel {
            context,
            classifier,
            output_labels,
            data_type,
        })
    }
}

#[derive(Debug, Error)]
pub enum ClassifierModelClassifyError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Classifier error: {0}")]
    Classifier(#[from] ClassifierEncodableError<B>),
    #[error("Input is empty")]
    EmptyInput,
    #[error("Input larger than model context size")]
    ContextOverflow,
}

impl<B: Backend> ClassifierModel<B> {
    pub fn classify(
        &self,
        input: &[u64],
    ) -> Result<HashMap<String, f32>, ClassifierModelClassifyError<B>> {
        if input.is_empty() {
            return Err(ClassifierModelClassifyError::EmptyInput);
        }

        if self.classifier.max_context_length().is_some_and(|max_context_length| input.len() > max_context_length) {
            return Err(ClassifierModelClassifyError::ContextOverflow);
        }

        let mut encoder = Encoder::<B>::new(&self.context).map_err(ClassifierModelClassifyError::Backend)?;

        let mut token_ids =
            encoder.allocate_constant(size_of_val(input)).map_err(ClassifierModelClassifyError::Backend)?;
        token_ids.copyin(input);

        let logits = self.classifier.encode(&token_ids, input.len(), &mut encoder)?;

        let mut output_buffer = self
            .context
            .create_allocation(self.output_labels.len() * self.data_type.size_in_bytes(), AllocationType::Global)
            .map_err(ClassifierModelClassifyError::Backend)?;

        encoder.encode_copy(&logits, .., &mut output_buffer, ..);

        drop(logits);
        drop(token_ids);

        encoder.end_encoding().submit().wait_until_completed().map_err(ClassifierModelClassifyError::Backend)?;

        assert!(self.data_type == DataType::BF16);

        let probs = output_buffer.copyout::<bf16>().into_iter().map(|logit| 1.0 / (1.0 + (-logit.to_f32()).exp()));

        Ok(self.output_labels.iter().cloned().zip(probs).collect())
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }
}
