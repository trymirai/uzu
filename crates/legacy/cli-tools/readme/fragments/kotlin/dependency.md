```kotlin
// settings.gradle.kts
includeBuild("<uzu checkout>/crates/legacy/uzu/bindings/kotlin")

// build.gradle.kts
dependencies {
    implementation("com.trymirai:uzu")
}
```

Build the engine for your machine first with `cargo tools build kotlin` in the uzu checkout. It runs on macOS (Apple
silicon, Metal), Linux and Windows (x86-64, Vulkan); the Vulkan requirements are in the
[CLI section](https://github.com/trymirai/uzu#cli) of the main README. The structured output example also uses
`org.json:json`.
On Windows, use a JDK whose bundled C++ runtime (`bin\msvcp140.dll`) is 14.40 or newer, such as Eclipse Temurin 21:
GPU drivers crash with an older one, like the 14.29 runtime of Android Studio's JDK.
