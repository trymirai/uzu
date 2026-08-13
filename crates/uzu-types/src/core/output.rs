use crate::core::content::ContentPart;

/// Output produced by an inference operation.
///
/// Implementations yield [`ContentPart`] values through their iterator and
/// provide a borrowed view of the content held by the output.
pub trait InferenceOutput: Iterator<Item = ContentPart> {
    fn content(&self) -> &[ContentPart];
}
