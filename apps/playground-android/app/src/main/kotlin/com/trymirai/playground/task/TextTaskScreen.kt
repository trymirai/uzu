package com.trymirai.playground.task

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.input.rememberTextFieldState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.trymirai.playground.session.ReplyStatsView
import com.trymirai.playground.session.SessionState
import com.trymirai.playground.session.label
import com.trymirai.playground.ui.components.ActionButton
import com.trymirai.playground.ui.components.PlaygroundTopBar
import com.trymirai.playground.ui.components.PromptField

@Composable
fun TextTaskScreen(identifier: String, viewModel: TextTaskViewModel, onBack: () -> Unit) {
    val task = viewModel.task
    val state by viewModel.state.collectAsStateWithLifecycle()
    val result by viewModel.result.collectAsStateWithLifecycle()
    val input = rememberTextFieldState(task.sample)
    val generating = state == SessionState.Generating
    Scaffold(topBar = { PlaygroundTopBar(onBack, identifier) }) { padding ->
        Column(Modifier.fillMaxSize().padding(padding).imePadding()) {
            Column(
                verticalArrangement = Arrangement.spacedBy(20.dp),
                modifier = Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(16.dp),
            ) {
                PromptField(input, enabled = state.canStart, placeholder = task.placeholder, maxLines = 10)
                (state as? SessionState.Failed)?.let { Text("Error: ${it.message}", color = MaterialTheme.colorScheme.error) }
                result?.let { reply ->
                    Column(Modifier.padding(start = 16.dp)) {
                        SelectionContainer {
                            Text(reply.message.text().orEmpty(), style = MaterialTheme.typography.titleMedium)
                        }
                        ReplyStatsView(reply.stats)
                    }
                }
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            ActionButton(
                text = state.label(idle = task.action, generating = "STOP"),
                enabled = generating || (state.canStart && input.text.isNotBlank()),
                onClick = { if (generating) viewModel.stop() else viewModel.run(input.text.toString()) },
                modifier = Modifier.padding(16.dp),
            )
        }
    }
}
