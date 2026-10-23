package com.trymirai.playground.chat

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.input.clearText
import androidx.compose.foundation.text.input.rememberTextFieldState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.trymirai.playground.R
import com.trymirai.playground.session.ReplyStatsView
import com.trymirai.playground.session.SessionState
import com.trymirai.playground.ui.components.PlaygroundTopBar
import com.trymirai.playground.ui.components.PromptField

@Composable
fun ChatScreen(identifier: String, viewModel: ChatViewModel, onSend: () -> Unit, onBack: () -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val messages by viewModel.messages.collectAsStateWithLifecycle()
    val listState = rememberLazyListState()
    // The list is laid out from the bottom, so a reply that grows while streaming keeps its latest text in view.
    // A new turn brings the conversation back to the bottom.
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) listState.animateScrollToItem(0)
    }
    Scaffold(topBar = { PlaygroundTopBar(onBack, identifier) }) { padding ->
        Column(Modifier.fillMaxSize().padding(padding).imePadding()) {
            LazyColumn(
                state = listState,
                reverseLayout = true,
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp, Alignment.Bottom),
                modifier = Modifier.weight(1f),
            ) {
                items(messages.asReversed()) { MessageRow(it) }
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Composer(
                state,
                onSend = { text ->
                    onSend()
                    viewModel.send(text)
                },
                onStop = viewModel::stop,
            )
        }
    }
}

@Composable
private fun Composer(state: SessionState, onSend: (String) -> Unit, onStop: () -> Unit) {
    val input = rememberTextFieldState()
    val canSend = state == SessionState.Idle && input.text.isNotBlank()
    val placeholder = when (state) {
        SessionState.Loading -> "Loading..."
        SessionState.Generating -> "Generating..."
        is SessionState.Failed -> "Error: ${state.message}"
        SessionState.Idle -> "Message..."
    }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .padding(16.dp)
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surfaceContainer, RoundedCornerShape(8.dp))
            .padding(end = 12.dp),
    ) {
        PromptField(
            input,
            enabled = state == SessionState.Idle,
            placeholder = placeholder,
            modifier = Modifier.weight(1f),
            container = Color.Transparent,
        )
        val generating = state == SessionState.Generating
        IconButton(
            onClick = {
                if (generating) {
                    onStop()
                } else {
                    onSend(input.text.toString())
                    input.clearText()
                }
            },
            enabled = generating || canSend,
            shape = RoundedCornerShape(8.dp),
            colors = IconButtonDefaults.iconButtonColors(
                containerColor = MaterialTheme.colorScheme.primary,
                contentColor = MaterialTheme.colorScheme.onPrimary,
                disabledContainerColor = MaterialTheme.colorScheme.outlineVariant,
                disabledContentColor = MaterialTheme.colorScheme.onSurfaceVariant,
            ),
            modifier = Modifier.size(28.dp),
        ) {
            Icon(
                painterResource(if (generating) R.drawable.ic_stop_filled else R.drawable.ic_arrow_upward),
                contentDescription = if (generating) "Stop" else "Send",
                modifier = Modifier.size(18.dp),
            )
        }
    }
}

@Composable
private fun MessageRow(message: Message) {
    val bubble = Modifier
        .background(MaterialTheme.colorScheme.surfaceContainer, RoundedCornerShape(8.dp))
        .padding(horizontal = 16.dp, vertical = 12.dp)
    if (message.isUser) {
        Box(Modifier.fillMaxWidth().padding(start = 64.dp), contentAlignment = Alignment.CenterEnd) {
            SelectionContainer { Text(message.content, style = MaterialTheme.typography.bodyLarge, modifier = bubble) }
        }
    } else {
        Column(Modifier.fillMaxWidth().padding(end = 64.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            SelectionContainer {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    message.reasoning?.let {
                        Text(
                            it,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = bubble,
                        )
                    }
                    Text(message.content, style = MaterialTheme.typography.bodyLarge)
                }
            }
            message.stats?.let { ReplyStatsView(it) }
        }
    }
}
