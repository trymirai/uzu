```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        // The JVM half of rustls-platform-verifier, which checks the engine's HTTPS certificates.
        maven("https://github.com/rustls/rustls-platform-verifier/raw/maven-archive/android-release-support/maven/")
    }
}
include(":uzu")
project(":uzu").projectDir = file("<uzu checkout>/crates/legacy/uzu/bindings/kotlin")

// app/build.gradle.kts
dependencies {
    implementation(project(":uzu"))
}
```

Building the module compiles the engine for arm64 with `cargo tools build kotlin`, which needs the Android NDK and
cargo-ndk (`cargo tools install kotlin`). Call `Uzu.initialize(context)` once before creating the engine.
