use enumset::EnumSetType;

/// Full or part of content
pub enum ContentPart {
    Text(TextContentPart),
    // Audio, Image, etc
}

pub struct TextContentPart {
    pub text: String,
}

#[derive(EnumSetType)]
pub enum ContentType {
    Text,
}
