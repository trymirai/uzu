# Mirai Playground for Android

The Android counterpart of [the iOS playground](../playground): chat, classification, summarization and text to
speech on models that run on the device, plus a catalog to download and delete them. It uses the Kotlin SDK in
[`crates/legacy/uzu/bindings/kotlin`](../../crates/legacy/uzu/bindings/kotlin) as a Gradle module.

## Requirements

- A phone with Vulkan 1.4 and arm64, such as a Pixel 10
- Android SDK with platform 37 and NDK 30.0.16248370 (`sdkmanager "platforms;android-37" "ndk;30.0.16248370"`)
- JDK 17 or newer (Android Studio's bundled JBR works)
- rustup with the `aarch64-linux-android` target and cargo-ndk: `cargo tools install kotlin`

## Run

Connect the phone over adb, then:

```bash
./gradlew installDebug
```

Every build first runs `cargo tools build kotlin release --targets android`, which compiles the engine for arm64 and
regenerates the Kotlin bindings. The first build compiles the whole engine and its shaders; later ones are incremental.

The SDK's instrumented tests and examples run on the connected phone:

```bash
cargo tools test kotlin
cargo tools example kotlin chat
```
