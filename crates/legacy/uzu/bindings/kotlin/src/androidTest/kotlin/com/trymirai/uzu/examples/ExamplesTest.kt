package com.trymirai.uzu.examples

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Test
import org.junit.runner.RunWith

/** Runs one example per method; `cargo tools example kotlin <name>` selects the method. */
@RunWith(AndroidJUnit4::class)
class ExamplesTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun quickStart() = runBlocking { runQuickStart(context) }

    @Test
    fun chat() = runBlocking { runChat(context) }

    @Test
    fun chatCloud() = runBlocking { runChatCloud(context) }

    @Test
    fun chatSharedInstance() = runBlocking { runChatSharedInstance(context) }

    @Test
    fun chatStructuredOutput() = runBlocking { runChatStructuredOutput(context) }

    @Test
    fun toolCalls() = runBlocking { runToolCalls(context) }
}
