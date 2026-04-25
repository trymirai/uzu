use uzu::{
    engine::{Engine, EngineConfig},
    types::{
        basic::{ReasoningEffort, SamplingMethod},
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig, ChatSpeculationPreset},
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("Qwen/Qwen3-0.6B".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let text_to_summarize = "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. \
        It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. \
        LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. \
        These models have a wide range of applications, including chatbots, content creation, translation, and code generation. \
        One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. \
        They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. \
        Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. \
        As these models grow in size and sophistication, they continue to enhance human-computer interactions, \
        making AI-powered communication more natural and effective.";
    let prompt = format!("Text is: \"{text_to_summarize}\". Write only summary itself.");
    let messages = vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::user().with_text(prompt),
    ];

    let chat_config = ChatConfig::default().with_speculation_preset(Some(ChatSpeculationPreset::Summarization {}));
    let session = engine.chat(model, chat_config).await?;

    let chat_reply_config =
        ChatReplyConfig::default().with_token_limit(Some(256)).with_sampling_method(SamplingMethod::Greedy {});
    let replies = session.reply(messages, chat_reply_config).await?;
    if let Some(reply) = replies.first() {
        println!("Summary: {}", reply.message.text().unwrap_or_default());
        println!("Generation t/s: {}", reply.stats.generate_tokens_per_second.unwrap_or_default());
    }

    Ok(())
}
