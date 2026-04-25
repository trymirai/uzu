use uzu::{
    engine::{Engine, EngineConfig},
    types::{
        basic::{Feature, ReasoningEffort, SamplingMethod},
        session::chat::{ChatConfig, ChatMessage, ChatReplyConfig, ChatSpeculationPreset},
    },
};

async fn quick_start_example() -> Result<(), Box<dyn std::error::Error>> {
    // snippet:quick-start
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("Qwen/Qwen3-0.6B".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    let messages = vec![
        ChatMessage::system().with_text("You are a helpful assistant".to_string()),
        ChatMessage::user().with_text("Tell me a short, funny story about a robot".to_string()),
    ];

    let session = engine.chat(model, ChatConfig::default()).await?;
    let replies = session.reply(messages, ChatReplyConfig::default()).await?;
    let message = replies.first().map(|reply| reply.message.clone());
    // endsnippet:quick-start

    if let Some(message) = message {
        println!("Reasoning: {}", message.reasoning().unwrap_or_default());
        println!("Text: {}", message.text().unwrap_or_default());
    }

    Ok(())
}

async fn chat_example() -> Result<(), Box<dyn std::error::Error>> {
    // snippet:engine-create
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;
    // endsnippet:engine-create

    // snippet:model-choose
    let model = engine.model("Qwen/Qwen3-0.6B".to_string()).await?.ok_or("Model not found")?;
    // endsnippet:model-choose

    // snippet:model-download
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }
    // endsnippet:model-download

    // snippet:session-create-general
    let session = engine.chat(model, ChatConfig::default()).await?;
    // endsnippet:session-create-general

    // snippet:session-input-general
    let messages = vec![
        ChatMessage::system().with_text("You are a helpful assistant".to_string()),
        ChatMessage::user().with_text("Tell me a short, funny story about a robot".to_string()),
    ];
    // endsnippet:session-input-general

    // snippet:session-run-general
    let replies = session.reply(messages, ChatReplyConfig::default()).await?;
    let message = replies.first().map(|reply| reply.message.clone());
    // endsnippet:session-run-general

    if let Some(message) = message {
        println!("Reasoning: {}", message.reasoning().unwrap_or_default());
        println!("Text: {}", message.text().unwrap_or_default());
    }

    Ok(())
}

async fn summarization_example() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("Qwen/Qwen3-0.6B".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    // snippet:session-input-summarization
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
    // endsnippet:session-input-summarization

    // snippet:session-create-summarization
    let chat_config = ChatConfig::default().with_speculation_preset(Some(ChatSpeculationPreset::Summarization {}));
    let session = engine.chat(model, chat_config).await?;
    // endsnippet:session-create-summarization

    // snippet:session-run-summarization
    let chat_reply_config =
        ChatReplyConfig::default().with_token_limit(Some(256)).with_sampling_method(SamplingMethod::Greedy {});
    let replies = session.reply(messages, chat_reply_config).await?;
    let reply = replies.first().cloned();
    // endsnippet:session-run-summarization

    if let Some(reply) = reply {
        println!("Summary: {}", reply.message.text().unwrap_or_default());
        println!("Generation t/s: {}", reply.stats.generate_tokens_per_second.unwrap_or_default());
    }

    Ok(())
}

async fn classification_example() -> Result<(), Box<dyn std::error::Error>> {
    let engine_config = EngineConfig::default();
    let engine = Engine::new(engine_config).await?;

    let model = engine.model("Qwen/Qwen3-0.6B".to_string()).await?.ok_or("Model not found")?;
    let downloader = engine.download(&model).await?;
    while let Some(update) = downloader.next().await {
        println!("Download progress: {}", update.progress());
    }

    // snippet:session-create-classification
    let feature = Feature {
        name: "sentiment".to_string(),
        values: vec![
            "Happy".to_string(),
            "Sad".to_string(),
            "Angry".to_string(),
            "Fearful".to_string(),
            "Surprised".to_string(),
            "Disgusted".to_string(),
        ],
    };
    let chat_config = ChatConfig::default().with_speculation_preset(Some(ChatSpeculationPreset::Classification {
        feature: feature.clone(),
    }));
    let session = engine.chat(model, chat_config).await?;
    // endsnippet:session-create-classification

    // snippet:session-input-classification
    let text_to_detect_feature = "Today's been awesome! Everything just feels right, and I can't stop smiling.";
    let prompt = format!(
        "Text is: \"{text_to_detect_feature}\". Choose {} from the list: {}. Answer with one word. Don't add a dot at the end.",
        feature.name,
        feature.values.join(", ")
    );
    let messages = vec![
        ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled),
        ChatMessage::user().with_text(prompt),
    ];
    // endsnippet:session-input-classification

    // snippet:session-run-classification
    let chat_reply_config =
        ChatReplyConfig::default().with_token_limit(Some(32)).with_sampling_method(SamplingMethod::Greedy {});
    let replies = session.reply(messages, chat_reply_config).await?;
    let reply = replies.first().cloned();
    // endsnippet:session-run-classification

    if let Some(reply) = reply {
        println!("Prediction: {}", reply.message.text().unwrap_or_default());
        println!("Generated tokens: {}", reply.stats.tokens_count_output.unwrap_or_default());
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    quick_start_example().await?;
    chat_example().await?;
    summarization_example().await?;
    classification_example().await?;
    Ok(())
}
