package com.trymirai.playground.session

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.trymirai.playground.ui.components.formatBytes
import com.trymirai.uzu.shoji.ChatReplyStats
import java.util.Locale

@Composable
fun ReplyStatsView(stats: ChatReplyStats) {
    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant, modifier = Modifier.padding(vertical = 4.dp))
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Metric("Time to first token:", "%.3f s".format(Locale.ROOT, stats.timeToFirstToken ?: 0.0))
        stats.generateTokensPerSecond?.takeIf { it > 0 }?.let {
            Metric("Tokens per second:", "%.3f t/s".format(Locale.ROOT, it))
        }
        Metric("Memory used:", stats.memoryUsedBytes?.let(::formatBytes) ?: "—")
        Metric("Total time:", "%.3f s".format(Locale.ROOT, stats.duration))
    }
}

@Composable
private fun Metric(label: String, value: String) {
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(label, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.labelMedium, maxLines = 1)
    }
}
