use uzu::{
    engine::Engine,
    session::chat::ChatSessionStreamChunk,
    types::{
        model::Model,
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
    },
};

pub fn is_placeholder(title: &str) -> bool {
    let n = title.trim().to_lowercase();
    n.is_empty() || n == "untitled" || n == "new chat"
}

pub fn sanitize(raw: &str) -> String {
    let mut title = raw.trim().replace(['\r', '\n'], " ");
    title = title.split_whitespace().collect::<Vec<_>>().join(" ");
    title.trim_matches(['"', '\'', '`']).trim().to_string()
}

pub fn prompt(user_message: &str) -> String {
    let quoted = user_message.split_whitespace().collect::<Vec<_>>().join(" ").replace('"', "\\\"");
    format!(
        "Generate a short, neutral chat title (2–4 words, no punctuation at the end) for the following user message. \
         If the topic is unclear, return \"General Chat\". Message: \"{quoted}\""
    )
}

pub fn vendor_fallback_title(model: &Model) -> Option<&'static str> {
    let vendor = model.family.as_ref().map(|f| f.vendor.name().to_lowercase())?;
    if vendor == "cartesia" || vendor == "liquidai" {
        Some("General Chat")
    } else {
        None
    }
}

pub async fn run(
    engine: &Engine,
    model: Model,
    user_message: &str,
) -> Result<String, String> {
    if let Some(fallback) = vendor_fallback_title(&model) {
        return Ok(fallback.to_string());
    }
    generate(engine, model, user_message).await
}

async fn generate(
    engine: &Engine,
    model: Model,
    user_message: &str,
) -> Result<String, String> {
    let session = engine.chat(model, ChatConfig::default()).await.map_err(|e| e.to_string())?;
    session.reset().await.map_err(|e| format!("{e:?}"))?;
    let messages = vec![ChatMessage::user().with_text(prompt(user_message))];
    let config = ChatReplyConfig::default().with_token_limit(Some(200));
    let stream = session.reply_with_stream(messages, config).await;
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            ChatSessionStreamChunk::Replies {
                replies,
            } => {
                if let Some(reply) = replies.into_iter().next() {
                    text = reply.message.text().unwrap_or_default();
                    if text.is_empty() {
                        text = reply.message.reasoning().unwrap_or_default();
                    }
                }
            },
            ChatSessionStreamChunk::Error {
                error,
            } => return Err(format!("{error:?}")),
        }
    }
    let candidate = sanitize(&text);
    Ok(if candidate.is_empty() {
        "General Chat".into()
    } else if candidate.chars().count() > 300 {
        "New chat".into()
    } else {
        candidate
    })
}
