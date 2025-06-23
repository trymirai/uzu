#[derive(Debug)]
pub enum GeneratorError {
    UnableToCreateMetalContext,
    UnableToLoadConfig,
    UnableToLoadWeights,
}
