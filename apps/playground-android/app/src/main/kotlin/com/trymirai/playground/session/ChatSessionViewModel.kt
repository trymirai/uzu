package com.trymirai.playground.session

import com.trymirai.playground.engine.EngineRepository
import com.trymirai.uzu.asFlow
import com.trymirai.uzu.nagare.ChatSession
import com.trymirai.uzu.nagare.ChatSessionStreamChunk
import com.trymirai.uzu.shoji.CancelToken
import com.trymirai.uzu.shoji.ChatMessage
import com.trymirai.uzu.shoji.ChatReply
import com.trymirai.uzu.shoji.ChatReplyConfig

/** Streams replies from one model's chat session, as chat, classification and summarization do. */
abstract class ChatSessionViewModel(identifier: String, engine: EngineRepository) :
    SessionViewModel<ChatSession>({ engine.chat(identifier) }) {
    private var cancelToken: CancelToken? = null

    /** Answers [input] from a fresh conversation, passing every partial reply to [onReply]. */
    protected fun reply(input: List<ChatMessage>, config: ChatReplyConfig, onReply: (ChatReply) -> Unit) =
        startGeneration { session ->
            session.reset()
            val stream = session.replyWithStream(input, config)
            cancelToken = stream.cancelToken()
            stream.asFlow().collect { chunk ->
                when (chunk) {
                    is ChatSessionStreamChunk.Replies -> chunk.replies.lastOrNull()?.let(onReply)
                    is ChatSessionStreamChunk.Error -> throw chunk.error
                    is ChatSessionStreamChunk.ToolResults -> Unit
                }
            }
        }

    /** Asks the engine to end the current reply; the state turns idle once the engine has finished the turn. */
    fun stop() {
        cancelToken?.cancel()
    }
}
