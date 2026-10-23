package com.trymirai.playground.home

import androidx.annotation.DrawableRes
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.trymirai.playground.Feature
import com.trymirai.playground.R
import com.trymirai.playground.ui.components.PlaygroundTopBar
import com.trymirai.playground.ui.components.ScreenHeader

private data class FeatureItem(@DrawableRes val icon: Int, val title: String, val feature: Feature?)

private val features = listOf(
    FeatureItem(R.drawable.ic_forum, "General Chat", Feature.Chat),
    FeatureItem(R.drawable.ic_contrast, "Classification", Feature.Classification),
    FeatureItem(R.drawable.ic_description, "Summarisation", Feature.Summarization),
    FeatureItem(R.drawable.ic_mic, "Text To Speech", Feature.TextToSpeech),
    FeatureItem(R.drawable.ic_photo_camera, "Camera", null),
)

private const val CONTACT_URL = "mailto:dima@getmirai.co,alexey@getmirai.co?subject=Interested%20in%20Mirai"

@Composable
fun HomeScreen(
    isPodcastPlaying: Boolean,
    onFeature: (Feature) -> Unit,
    onExploreModels: () -> Unit,
    onTogglePodcast: () -> Unit,
    onAbout: () -> Unit,
) {
    val uriHandler = LocalUriHandler.current
    Scaffold(topBar = { PlaygroundTopBar(onBack = null) }) { padding ->
        Column(Modifier.fillMaxSize().padding(padding)) {
            Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) {
                ScreenHeader("Choose AI Usecase", "to see Mirai on-device possibilities")
                Column(Modifier.padding(horizontal = 20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    features.forEach { item -> FeatureRow(item, onFeature) }
                }
            }
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier.fillMaxWidth().clickable(onClick = onExploreModels).padding(vertical = 20.dp),
            ) {
                Icon(painterResource(R.drawable.ic_code), contentDescription = null, modifier = Modifier.size(16.dp))
                Text(
                    "Explore our available models",
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(top = 8.dp, bottom = 16.dp),
                )
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
            }
            Row(
                horizontalArrangement = Arrangement.SpaceBetween,
                modifier = Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, bottom = 20.dp),
            ) {
                FooterLink(
                    if (isPodcastPlaying) R.drawable.ic_pause_filled else R.drawable.ic_play_arrow_filled,
                    "Our vision",
                    onTogglePodcast,
                )
                FooterLink(R.drawable.ic_info_filled, "About us", onAbout)
                FooterLink(R.drawable.ic_favorite_filled, "Contact us") { uriHandler.openUri(CONTACT_URL) }
            }
        }
    }
}

@Composable
private fun FeatureRow(item: FeatureItem, onFeature: (Feature) -> Unit) {
    val feature = item.feature
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surfaceContainer, RoundedCornerShape(12.dp))
            .clickable(enabled = feature != null) { feature?.let(onFeature) }
            .padding(16.dp),
    ) {
        Icon(
            painterResource(item.icon),
            contentDescription = null,
            tint = if (feature == null) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.primary,
        )
        Text(item.title, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(start = 16.dp).weight(1f))
        if (feature == null) {
            Text("COMING SOON", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun FooterLink(@DrawableRes icon: Int, title: String, onClick: () -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.clickable(onClick = onClick).padding(horizontal = 16.dp, vertical = 8.dp),
    ) {
        Icon(painterResource(icon), contentDescription = null, modifier = Modifier.size(16.dp))
        Text(title, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 8.dp))
    }
}
