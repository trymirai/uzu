package com.trymirai.playground.speech

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.input.rememberTextFieldState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.trymirai.playground.audio.AudioPlayer
import com.trymirai.playground.audio.AudioSource
import com.trymirai.playground.session.SessionState
import com.trymirai.playground.ui.components.ActionButton
import com.trymirai.playground.ui.components.PlaygroundTopBar
import com.trymirai.playground.ui.components.PromptField

private const val SAMPLE = "London is the capital of United Kingdom and one of the world’s most influential cities, " +
    "known for its rich history, cultural diversity, and global significance in finance, politics, and the arts. " +
    "Situated along the River Thames, the city blends historic landmarks like Tower of London and Buckingham Palace " +
    "with modern architecture such as The Shard. London is also home to renowned institutions including the British " +
    "Museum and vibrant areas like Covent Garden, offering a mix of history, entertainment, and innovation that " +
    "attracts millions of visitors each year."

@Composable
fun TextToSpeechScreen(identifier: String, viewModel: TextToSpeechViewModel, audio: AudioPlayer, onBack: () -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val output by viewModel.output.collectAsStateWithLifecycle()
    val playing by audio.playing.collectAsStateWithLifecycle()
    val input = rememberTextFieldState(SAMPLE)
    val speech = output?.let(AudioSource::File)
    Scaffold(topBar = { PlaygroundTopBar(onBack, identifier) }) { padding ->
        Column(Modifier.fillMaxSize().padding(padding).imePadding()) {
            Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(16.dp)) {
                PromptField(input, enabled = state == SessionState.Idle, placeholder = "Enter text for synthesis…")
            }
            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.padding(16.dp)) {
                ActionButton(
                    text = when (val current = state) {
                        SessionState.Loading -> "Loading…"
                        SessionState.Generating -> "GENERATING…"
                        is SessionState.Failed -> "Error: ${current.message}"
                        SessionState.Idle -> "GENERATE"
                    },
                    enabled = state == SessionState.Idle && input.text.isNotBlank(),
                    onClick = {
                        // The new synthesis overwrites the file the player may hold open.
                        audio.release()
                        viewModel.generate(input.text.toString())
                    },
                    modifier = Modifier.weight(1f),
                )
                ActionButton(
                    text = if (speech != null && playing == speech) "PAUSE" else "PLAY",
                    enabled = speech != null && state != SessionState.Generating,
                    onClick = { speech?.let(audio::toggle) },
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }
}
