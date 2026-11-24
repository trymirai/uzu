package com.trymirai.uzu.examples

import kotlinx.coroutines.runBlocking
import kotlin.test.Test

class ExamplesTest {
    @Test
    fun quickStart() = runBlocking { runQuickStart() }

    @Test
    fun chat() = runBlocking { runChat() }

    @Test
    fun chatCloud() = runBlocking { runChatCloud() }

    @Test
    fun chatSharedInstance() = runBlocking { runChatSharedInstance() }

    @Test
    fun chatStructuredOutput() = runBlocking { runChatStructuredOutput() }

    @Test
    fun toolCalls() = runBlocking { runToolCalls() }
}
