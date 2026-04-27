import Uzu

public func runChatForSummarization() async throws {
    let engineConfig = EngineConfig.create()
    let engine = try await Engine.create(config: engineConfig)
    
    guard let model = try await engine.model(identifier: "Qwen/Qwen3-0.6B") else {
        return
    }
    for try await update in try await engine.download(model: model).iterator() {
        print("Download progress: \(update.progress())")
    }
    
    let textToSummarize = "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates human-like text. It is trained on vast datasets containing books, articles, and web content, allowing it to understand and predict language patterns. LLMs use deep learning, particularly transformer-based architectures, to analyze text, recognize context, and generate coherent responses. These models have a wide range of applications, including chatbots, content creation, translation, and code generation. One of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving accuracy and fluency. Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's LLaMA. As these models grow in size and sophistication, they continue to enhance human-computer interactions, making AI-powered communication more natural and effective.";
    let prompt = "Text is: \"\(textToSummarize)\". Write only summary itself."
    let messages = [
        ChatMessage.system().withReasoningEffort(reasoningEffort: .disabled),
        ChatMessage.user().withText(text: prompt)
    ]
    
    let chatConfig = ChatConfig.create().withSpeculationPreset(speculationPreset: .summarization)
    let session = try await engine.chat(model: model, config: chatConfig)
    
    let chatReplyConfig = ChatReplyConfig.create().withTokenLimit(tokenLimit: 256).withSamplingMethod(samplingMethod: .greedy)
    let replies = try await session.reply(input: messages, config: chatReplyConfig)
    guard let reply = replies.last else {
        return
    }
    
    print("Summary: \(reply.message.text() ?? "empty")")
    print("Generation t\\s: \(reply.stats.generateTokensPerSecond ?? 0.0)")
}
