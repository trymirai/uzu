use openai_harmony::HarmonyEncodingName as ExternalEncodingName;

use crate::chat::harmony::EncodingName;

impl From<EncodingName> for ExternalEncodingName {
    fn from(encoding: EncodingName) -> ExternalEncodingName {
        match encoding {
            EncodingName::GptOss => ExternalEncodingName::HarmonyGptOss,
        }
    }
}

impl From<ExternalEncodingName> for EncodingName {
    fn from(encoding_name: ExternalEncodingName) -> EncodingName {
        match encoding_name {
            ExternalEncodingName::HarmonyGptOss => EncodingName::GptOss,
        }
    }
}
