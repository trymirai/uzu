package com.trymirai.playground.ui.components

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.input.TextFieldLineLimits
import androidx.compose.foundation.text.input.TextFieldState
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.trymirai.playground.R
import java.util.Locale

/** The top bar every screen shares: the Mirai logo, plus the model a feature screen runs. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PlaygroundTopBar(onBack: (() -> Unit)?, modelIdentifier: String? = null) {
    CenterAlignedTopAppBar(
        title = { Icon(painterResource(R.drawable.ic_mirai_logo), contentDescription = "Mirai") },
        navigationIcon = {
            if (onBack != null) {
                IconButton(onClick = onBack) {
                    Icon(painterResource(R.drawable.ic_arrow_back), contentDescription = "Back")
                }
            }
        },
        actions = {
            if (modelIdentifier != null) {
                Text(
                    text = modelIdentifier,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.MiddleEllipsis,
                    modifier = Modifier.fillMaxWidth(0.4f).padding(end = 12.dp),
                )
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
    )
}

@Composable
fun ScreenHeader(title: String, subtitle: String) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier.fillMaxWidth().padding(vertical = 28.dp),
    ) {
        Text(title.uppercase(), style = MaterialTheme.typography.titleMedium)
        Text(
            subtitle,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 8.dp),
        )
    }
}

@Composable
fun ActionButton(text: String, enabled: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Button(
        onClick = onClick,
        enabled = enabled,
        shape = RoundedCornerShape(12.dp),
        colors = ButtonDefaults.buttonColors(
            disabledContainerColor = MaterialTheme.colorScheme.surfaceContainer,
            disabledContentColor = MaterialTheme.colorScheme.onSurfaceVariant,
        ),
        modifier = modifier.fillMaxWidth().height(56.dp),
    ) {
        Text(text, style = MaterialTheme.typography.labelLarge, maxLines = 2, overflow = TextOverflow.Ellipsis)
    }
}

@Composable
fun PromptField(
    state: TextFieldState,
    enabled: Boolean,
    placeholder: String,
    modifier: Modifier = Modifier,
    maxLines: Int = Int.MAX_VALUE,
    container: Color = MaterialTheme.colorScheme.surfaceContainer,
) {
    TextField(
        state = state,
        enabled = enabled,
        placeholder = { Text(placeholder, style = MaterialTheme.typography.bodyLarge) },
        textStyle = MaterialTheme.typography.bodyLarge,
        lineLimits = TextFieldLineLimits.MultiLine(maxHeightInLines = maxLines),
        shape = RoundedCornerShape(8.dp),
        colors = TextFieldDefaults.colors(
            focusedContainerColor = container,
            unfocusedContainerColor = container,
            disabledContainerColor = container,
            focusedIndicatorColor = Color.Transparent,
            unfocusedIndicatorColor = Color.Transparent,
            disabledIndicatorColor = Color.Transparent,
        ),
        modifier = modifier.fillMaxWidth(),
    )
}

fun formatBytes(bytes: Long): String {
    val kilobyte = 1024.0
    val value = bytes.toDouble()
    return when {
        value >= kilobyte * kilobyte * kilobyte -> "%.1f GB".format(Locale.ROOT, value / (kilobyte * kilobyte * kilobyte))
        value >= kilobyte * kilobyte -> "%.1f MB".format(Locale.ROOT, value / (kilobyte * kilobyte))
        value >= kilobyte -> "%.0f KB".format(Locale.ROOT, value / kilobyte)
        else -> "$bytes B"
    }
}
