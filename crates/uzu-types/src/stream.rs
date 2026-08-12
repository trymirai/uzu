use crate::content::Content;

trait UzuStream: Iterator<Item = Content> {
    fn content(&self) -> &[Content];
}
