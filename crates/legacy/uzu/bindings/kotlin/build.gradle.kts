// An Android library for any Gradle build to include; the including build supplies the Android Gradle plugin.
plugins {
    id("com.android.library")
}

android {
    namespace = "com.trymirai.uzu"
    compileSdk = 37

    defaultConfig {
        // The API level the engine is linked against: CARGO_NDK_PLATFORM in .config/platforms.toml.
        minSdk = 31
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }
}

androidComponents {
    onVariants { variant ->
        variant.sources.kotlin?.addStaticSourceDirectory("src/main/generated")
    }
}

// `cargo tools build kotlin` compiles libuzu.so into src/main/jniLibs and writes its UniFFI bindings into
// src/main/generated. The engine is always built optimized; app variants differ only in Kotlin code.
val buildNativeLibrary by tasks.registering(Exec::class) {
    description = "Builds the native engine and its Kotlin bindings"
    commandLine("cargo", "tools", "build", "kotlin", "release", "--targets", "android")
}

tasks.named("preBuild") {
    dependsOn(buildNativeLibrary)
}

// The JVM half of rustls-platform-verifier must match the crate version the native library links.
val rustlsPlatformVerifier = providers.fileContents(layout.projectDirectory.file("../../../../../Cargo.lock")).asText.map {
    val version = Regex("""name = "rustls-platform-verifier-android"\s+version = "([^"]+)"""").find(it)?.groupValues?.get(1)
        ?: error("rustls-platform-verifier-android is missing from Cargo.lock")
    "org.rustls:rustls-platform-verifier:$version"
}

dependencies {
    api("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")
    implementation("androidx.annotation:annotation:1.11.0")
    implementation("net.java.dev.jna:jna:5.19.1@aar")
    addProvider("implementation", rustlsPlatformVerifier)

    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
}
