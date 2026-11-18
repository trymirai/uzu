package com.trymirai.playground.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

// The iOS playground's monochrome palette: primary text on a background, cards one step off it.
private val LightColors = lightColorScheme(
    primary = Color.Black,
    onPrimary = Color.White,
    background = Color.White,
    onBackground = Color.Black,
    surface = Color.White,
    onSurface = Color.Black,
    onSurfaceVariant = Color(0x993C3C43),
    surfaceContainer = Color(0xFFF4F5F5),
    outlineVariant = Color(0xFFE5E5E5),
    error = Color(0xFFFF3B30),
)

private val DarkColors = darkColorScheme(
    primary = Color.White,
    onPrimary = Color.Black,
    background = Color.Black,
    onBackground = Color.White,
    surface = Color.Black,
    onSurface = Color.White,
    onSurfaceVariant = Color(0x99EBEBF5),
    surfaceContainer = Color(0xFF1C1C1E),
    outlineVariant = Color(0xFF38383A),
    error = Color(0xFFFF453A),
)

private fun mono(size: Int, weight: FontWeight = FontWeight.Normal) =
    TextStyle(fontFamily = FontFamily.Monospace, fontSize = size.sp, fontWeight = weight)

private val MonospacedTypography = Typography(
    titleMedium = mono(16, FontWeight.SemiBold),
    bodyLarge = mono(16),
    bodyMedium = mono(14),
    bodySmall = mono(12),
    labelLarge = mono(14, FontWeight.Bold),
    labelMedium = mono(12, FontWeight.SemiBold),
    labelSmall = mono(10, FontWeight.SemiBold),
)

@Composable
fun PlaygroundTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) DarkColors else LightColors,
        typography = MonospacedTypography,
        content = content,
    )
}
