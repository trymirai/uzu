use serde::Deserialize;

const IMAGE_URL_SCHEMES: [&str; 3] = ["http://", "https://", "data:"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OaiContentPart {
    Text(String),
    Image(String),
}

pub fn deserialize_message_content<'de, D>(deserializer: D) -> Result<Vec<OaiContentPart>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match Option::<serde_json::Value>::deserialize(deserializer)? {
        None => Ok(Vec::new()),
        Some(serde_json::Value::String(text)) => Ok(vec![OaiContentPart::Text(text)]),
        Some(serde_json::Value::Array(parts)) => {
            let mut content = Vec::new();
            for part in &parts {
                match part.get("type").and_then(serde_json::Value::as_str) {
                    Some("text") => {
                        let text = part.get("text").and_then(serde_json::Value::as_str).unwrap_or_default();
                        // Adjacent text parts form one text block.
                        match content.last_mut() {
                            Some(OaiContentPart::Text(previous)) => previous.push_str(text),
                            _ => content.push(OaiContentPart::Text(text.to_string())),
                        }
                    },
                    Some("image_url") => {
                        let url = part
                            .get("image_url")
                            .and_then(|image_url| image_url.get("url"))
                            .and_then(serde_json::Value::as_str)
                            .ok_or_else(|| {
                                serde::de::Error::custom("image_url content part is missing image_url.url")
                            })?;
                        // The engine reads any other string as a local file path, which a client must not reach.
                        if !IMAGE_URL_SCHEMES.iter().any(|scheme| url.starts_with(scheme)) {
                            return Err(serde::de::Error::custom("image_url.url must be an http(s) or data URL"));
                        }
                        content.push(OaiContentPart::Image(url.to_string()));
                    },
                    Some(other) => {
                        return Err(serde::de::Error::custom(format!(
                            "unsupported message content part type: {other}"
                        )));
                    },
                    None => return Err(serde::de::Error::custom("message content part is missing a type")),
                }
            }
            Ok(content)
        },
        Some(_) => Err(serde::de::Error::custom("message content must be a string or an array of content parts")),
    }
}
