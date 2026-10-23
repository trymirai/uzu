package com.trymirai.playground

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.viewmodel.navigation3.rememberViewModelStoreNavEntryDecorator
import androidx.navigation3.runtime.entryProvider
import androidx.navigation3.runtime.rememberNavBackStack
import androidx.navigation3.runtime.rememberSaveableStateHolderNavEntryDecorator
import androidx.navigation3.ui.NavDisplay
import com.trymirai.playground.about.AboutScreen
import com.trymirai.playground.audio.AudioSource
import com.trymirai.playground.chat.ChatScreen
import com.trymirai.playground.chat.ChatViewModel
import com.trymirai.playground.home.HomeScreen
import com.trymirai.playground.models.ModelListScreen
import com.trymirai.playground.speech.TextToSpeechScreen
import com.trymirai.playground.speech.TextToSpeechViewModel
import com.trymirai.playground.task.TextTask
import com.trymirai.playground.task.TextTaskScreen
import com.trymirai.playground.task.TextTaskViewModel

private val podcast = AudioSource.Asset("about_us.mp3")

@Composable
fun PlaygroundApp() {
    val application = LocalContext.current.applicationContext as PlaygroundApplication
    val playing by application.audio.playing.collectAsStateWithLifecycle()
    val backStack = rememberNavBackStack(Route.Home)
    val back: () -> Unit = { backStack.removeLastOrNull() }

    NavDisplay(
        backStack = backStack,
        onBack = back,
        entryDecorators = listOf(
            rememberSaveableStateHolderNavEntryDecorator(),
            rememberViewModelStoreNavEntryDecorator(),
        ),
        entryProvider = entryProvider {
            entry<Route.Home> {
                HomeScreen(
                    isPodcastPlaying = playing == podcast,
                    onFeature = { backStack.add(Route.ModelSelection(it)) },
                    onExploreModels = { backStack.add(Route.ModelManagement) },
                    onTogglePodcast = { application.audio.toggle(podcast) },
                    onAbout = { backStack.add(Route.About) },
                )
            }
            entry<Route.ModelSelection> { route ->
                ModelListScreen(route.next, application.engine, { backStack.add(route.next.route(it)) }, back)
            }
            entry<Route.ModelManagement> {
                ModelListScreen(null, application.engine, onChoose = {}, onBack = back)
            }
            entry<Route.Chat> { route ->
                ChatScreen(
                    route.identifier,
                    viewModel { ChatViewModel(route.identifier, application.engine) },
                    onSend = application.audio::pause,
                    onBack = back,
                )
            }
            entry<Route.Classification> { route ->
                TextTaskScreen(
                    route.identifier,
                    TextTask.Classification,
                    viewModel { TextTaskViewModel(route.identifier, application.engine, TextTask.Classification) },
                    back,
                )
            }
            entry<Route.Summarization> { route ->
                TextTaskScreen(
                    route.identifier,
                    TextTask.Summarization,
                    viewModel { TextTaskViewModel(route.identifier, application.engine, TextTask.Summarization) },
                    back,
                )
            }
            entry<Route.TextToSpeech> { route ->
                TextToSpeechScreen(
                    route.identifier,
                    viewModel {
                        TextToSpeechViewModel(
                            route.identifier,
                            application.engine,
                            application.cacheDir.resolve("tts_output.wav").absolutePath,
                        )
                    },
                    application.audio,
                    back,
                )
            }
            entry<Route.About> { AboutScreen(back) }
        },
    )
}
