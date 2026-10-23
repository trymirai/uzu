package com.trymirai.playground.models

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.trymirai.playground.Feature
import com.trymirai.playground.R
import com.trymirai.playground.engine.EngineRepository
import com.trymirai.playground.ui.components.ActionButton
import com.trymirai.playground.ui.components.PlaygroundTopBar
import com.trymirai.playground.ui.components.ScreenHeader
import com.trymirai.playground.ui.components.formatBytes
import com.trymirai.uzu.downloadmanager.DownloadPhase
import com.trymirai.uzu.downloadmanager.DownloadState
import com.trymirai.uzu.shoji.Model
import com.trymirai.uzu.shoji.ModelSpecialization

private enum class Section(val title: String) {
    Installed("Installed"),
    Installing("Installing"),
    Paused("Paused"),
    NotInstalled("Not Installed"),
}

private fun DownloadState.section() = when (phase) {
    DownloadPhase.Downloaded -> Section.Installed
    DownloadPhase.Downloading, DownloadPhase.Initializing, is DownloadPhase.Locked -> Section.Installing
    DownloadPhase.Paused -> Section.Paused
    DownloadPhase.NotDownloaded, is DownloadPhase.Error -> Section.NotInstalled
}

/** Picks a model for [feature], or manages downloads when [feature] is null. */
@Composable
fun ModelListScreen(feature: Feature?, engine: EngineRepository, onChoose: (String) -> Unit, onBack: () -> Unit) {
    val models by engine.models.collectAsStateWithLifecycle()
    val states by engine.downloadStates.collectAsStateWithLifecycle()
    val failure by engine.failure.collectAsStateWithLifecycle()
    var selected by rememberSaveable { mutableStateOf<String?>(null) }

    val available = models.filter { model ->
        when (feature) {
            null -> true
            Feature.TextToSpeech -> ModelSpecialization.TEXT_TO_SPEECH in model.specializations
            else -> ModelSpecialization.CHAT in model.specializations
        }
    }
    val cloud = if (feature == Feature.Chat) available.filter { it.isRemote() }.sortedBy { it.name() } else emptyList()
    val local = available
        .filter { it.isOnDevice() }
        .mapNotNull { model -> states[model.identifier]?.let { model to it } }
        .sortedBy { (model, _) -> model.identifier }
        .groupBy { (_, state) -> state.section() }
    val selectedModel = available.firstOrNull { it.identifier == selected }

    Scaffold(
        topBar = { PlaygroundTopBar(onBack) },
        bottomBar = {
            Column(Modifier.navigationBarsPadding().padding(20.dp)) {
                failure?.let {
                    Text(
                        it,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error,
                        modifier = Modifier.padding(bottom = 8.dp),
                    )
                }
                BottomAction(feature, selectedModel, selectedModel?.let { states[it.identifier] }, engine, onChoose)
            }
        },
    ) { padding ->
        LazyColumn(
            contentPadding = PaddingValues(start = 20.dp, end = 20.dp, bottom = 20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier.fillMaxSize().padding(padding),
        ) {
            item { ScreenHeader("Available AI Models", "select a local model to install") }
            if (cloud.isNotEmpty()) {
                item { SectionTitle("Cloud") }
                items(cloud, key = { it.identifier }) { model ->
                    CloudModelRow(model, model.identifier == selected, Modifier.animateItem()) {
                        selected = model.identifier
                    }
                }
            }
            Section.entries.forEach { section ->
                val rows = local[section] ?: return@forEach
                item(key = section) { SectionTitle(section.title) }
                items(rows, key = { (model, _) -> model.identifier }) { (model, state) ->
                    ModelRow(model, state, model.identifier == selected, Modifier.animateItem()) {
                        selected = model.identifier
                    }
                }
            }
        }
    }
}

@Composable
private fun BottomAction(
    feature: Feature?,
    model: Model?,
    state: DownloadState?,
    engine: EngineRepository,
    onChoose: (String) -> Unit,
) {
    val phase = state?.phase
    when {
        model != null && model.isRemote() && feature == Feature.Chat ->
            ActionButton("CHOOSE CLOUD MODEL", enabled = true, onClick = { onChoose(model.identifier) })
        model == null || !model.isOnDevice() || phase == null ->
            ActionButton(if (feature == null) "SELECT MODEL" else "CHOOSE", enabled = false, onClick = {})
        phase == DownloadPhase.Downloaded -> if (feature == null) {
            ActionButton("DELETE", enabled = true, onClick = { engine.delete(model) })
        } else {
            ActionButton("CHOOSE", enabled = true, onClick = { onChoose(model.identifier) })
        }
        phase == DownloadPhase.Downloading && feature == null ->
            ActionButton("PAUSE", enabled = true, onClick = { engine.pause(model) })
        phase == DownloadPhase.Downloading || phase == DownloadPhase.Initializing || phase is DownloadPhase.Locked ->
            Text(
                "Wait for the models to be installed…",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth().padding(vertical = 18.dp),
            )
        phase == DownloadPhase.Paused ->
            ActionButton(if (feature == null) "RESUME" else "DOWNLOAD", enabled = true, onClick = { engine.download(model) })
        phase is DownloadPhase.Error && feature != null ->
            ActionButton("RETRY DOWNLOAD", enabled = true, onClick = { engine.download(model) })
        else -> ActionButton("DOWNLOAD", enabled = true, onClick = { engine.download(model) })
    }
}

@Composable
private fun SectionTitle(title: String) {
    Text(
        title.uppercase(),
        style = MaterialTheme.typography.labelMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier.padding(start = 4.dp, top = 12.dp),
    )
}

@Composable
private fun ModelRow(model: Model, state: DownloadState, selected: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val phase = state.phase
    Column(
        verticalArrangement = Arrangement.spacedBy(8.dp),
        modifier = modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surfaceContainer, RoundedCornerShape(12.dp))
            .clickable(onClick = onClick)
            .padding(16.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Icon(
                painterResource(if (selected) R.drawable.ic_check_circle_filled else R.drawable.ic_radio_button_unchecked),
                contentDescription = null,
                modifier = Modifier.size(24.dp),
            )
            Column(verticalArrangement = Arrangement.spacedBy(4.dp), modifier = Modifier.weight(1f)) {
                Text(model.name(), style = MaterialTheme.typography.bodyMedium, maxLines = 2)
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    model.family?.vendor?.name()?.let { Badge(it) }
                    model.quantization?.name()?.let { Badge(it) }
                }
            }
            Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    when (phase) {
                        DownloadPhase.NotDownloaded -> "not installed"
                        DownloadPhase.Initializing, DownloadPhase.Downloading -> "installing…"
                        DownloadPhase.Downloaded -> "installed"
                        DownloadPhase.Paused -> "paused"
                        is DownloadPhase.Error -> "error"
                        is DownloadPhase.Locked -> "installing elsewhere…"
                    },
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (phase is DownloadPhase.Error) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
                )
                when (phase) {
                    DownloadPhase.Downloading, DownloadPhase.Paused, is DownloadPhase.Locked ->
                        "${formatBytes(state.downloadedBytes)} / ${formatBytes(state.totalBytes)}"
                    DownloadPhase.NotDownloaded -> formatBytes(state.totalBytes)
                    else -> null
                }?.let {
                    Text(it, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
        if (state.isInProgress() || phase == DownloadPhase.Paused) {
            LinearProgressIndicator(
                progress = { state.progress().coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth().height(4.dp),
            )
        }
    }
}

@Composable
private fun CloudModelRow(model: Model, selected: Boolean, modifier: Modifier, onClick: () -> Unit) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surfaceContainer, RoundedCornerShape(12.dp))
            .clickable(onClick = onClick)
            .padding(16.dp),
    ) {
        Icon(painterResource(if (selected) R.drawable.ic_cloud_filled else R.drawable.ic_cloud), contentDescription = null)
        Column(verticalArrangement = Arrangement.spacedBy(4.dp), modifier = Modifier.weight(1f)) {
            Text(model.name(), style = MaterialTheme.typography.bodyMedium)
            Badge(model.family?.vendor?.name() ?: model.registry.name())
        }
        Text("cloud", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun Badge(text: String) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier
            .background(MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(4.dp))
            .padding(horizontal = 6.dp, vertical = 2.dp),
    )
}
