package com.trymirai.playground.task

private val sentiments = listOf("Happy", "Sad", "Angry", "Fearful", "Surprised", "Disgusted")

/** A one-shot prompt over a piece of text, answered greedily without reasoning. */
enum class TextTask(val placeholder: String, val action: String, val sample: String, val tokenLimit: UInt) {
    Classification(
        placeholder = "Enter text to classify…",
        action = "CLASSIFY",
        sample = "Today's been awesome! Everything just feels right, and I can't stop smiling.",
        tokenLimit = 32u,
    ),
    Summarization(
        placeholder = "Enter text to summarise…",
        action = "SUMMARISE",
        sample = "A Large Language Model (LLM) is a type of artificial intelligence that processes and generates " +
            "human-like text. It is trained on vast datasets containing books, articles, and web content, allowing it " +
            "to understand and predict language patterns. LLMs use deep learning, particularly transformer-based " +
            "architectures, to analyze text, recognize context, and generate coherent responses. These models have a " +
            "wide range of applications, including chatbots, content creation, translation, and code generation. One " +
            "of the key strengths of LLMs is their ability to generate contextually relevant text based on prompts. " +
            "They utilize self-attention mechanisms to weigh the importance of words within a sentence, improving " +
            "accuracy and fluency. Examples of popular LLMs include OpenAI's GPT series, Google's BERT, and Meta's " +
            "LLaMA. As these models grow in size and sophistication, they continue to enhance human-computer " +
            "interactions, making AI-powered communication more natural and effective.",
        tokenLimit = 1024u,
    );

    fun prompt(text: String): String = when (this) {
        Classification -> "Text is: \"$text\". Choose sentiment from the list: ${sentiments.joinToString()}. " +
            "Answer with one word. Don't add dot at the end."
        Summarization -> "Text is: \"$text\". Write only summary itself."
    }
}
