# Mirai Playground for Android

The Android counterpart of [the iOS playground](../playground): chat, classification, summarization and text to
speech on on-device (and, for chat, cloud) models, plus a catalog to download and delete them. It uses the Kotlin SDK in
[`crates/legacy/uzu/bindings/kotlin`](../../crates/legacy/uzu/bindings/kotlin) as a Gradle module.

## Requirements

- A phone with Vulkan 1.4 and arm64, such as a Pixel 10
- Android SDK with platform 37 and NDK 30.0.16248370 (`sdkmanager "platforms;android-37" "ndk;30.0.16248370"`)
- JDK 17 or newer (Android Studio's bundled JBR works)
- rustup with the `aarch64-linux-android` target and cargo-ndk: `cargo tools install kotlin`
- Slang, which compiles the Vulkan shaders, and libclang, which bindgen uses for Slang's bindings: set `VULKAN_SDK` or
  `SLANG_DIR`, and `LIBCLANG_PATH` if bindgen cannot find libclang. On macOS, also set
  `SDKROOT=$(xcrun --show-sdk-path)` and add Slang's `lib` directory to `DYLD_FALLBACK_LIBRARY_PATH`

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

Their output goes to the phone's log, not the terminal: follow it with `adb logcat -s System.out`, or add `-d` to dump
it afterwards.
