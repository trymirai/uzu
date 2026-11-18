package com.trymirai.playground.about

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.unit.dp
import com.trymirai.playground.ui.components.PlaygroundTopBar

private data class Link(val url: String, val title: String, val subtitle: String? = null)

private val links = listOf(
    Link(
        "https://trymirai.com",
        "Mirai",
        "AI which run directly on your devices, bringing powerful capabilities closer to where decisions are made.",
    ),
    Link(
        "https://saikollm.com",
        "Saiko",
        "A family of small AI models, highly optimized for on-device tasks. Zero cloud dependency, instant response times.",
    ),
    Link("https://trymirai.com/blog/deploying-llms-on-mobile", "Introduction to Deploying LLMs on Mobile"),
    Link("https://trymirai.com/blog/how-to-understand-on-device-ai", "How to Understand On-Device AI"),
    Link("https://trymirai.com/blog/brief-history-of-apple-ml-stack", "Brief history of Apple ML Stack"),
)

@Composable
fun AboutScreen(onBack: () -> Unit) {
    val uriHandler = LocalUriHandler.current
    Scaffold(topBar = { PlaygroundTopBar(onBack) }) { padding ->
        LazyColumn(
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.fillMaxSize().padding(padding),
        ) {
            items(links) { link ->
                Column(
                    Modifier
                        .fillMaxWidth()
                        .background(MaterialTheme.colorScheme.surfaceContainer, RoundedCornerShape(16.dp))
                        .clickable { uriHandler.openUri(link.url) }
                        .padding(16.dp),
                ) {
                    Text(link.title, style = MaterialTheme.typography.bodyLarge)
                    link.subtitle?.let {
                        Text(
                            it,
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(top = 4.dp),
                        )
                    }
                }
            }
        }
    }
}
