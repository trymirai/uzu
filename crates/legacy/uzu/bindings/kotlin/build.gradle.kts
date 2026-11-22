plugins {
    id("com.android.library")
}

android {
    namespace = "com.trymirai.uzu"
    compileSdk = 37

    defaultConfig {
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

val buildNativeLibrary by tasks.registering(Exec::class) {
    description = "Builds the native engine and its Kotlin bindings"
    commandLine("cargo", "tools", "build", "kotlin", "release")
}

tasks.named("preBuild") {
    dependsOn(buildNativeLibrary)
}

val rustlsPlatformVerifier = providers.fileContents(layout.projectDirectory.file("../../../../../Cargo.lock")).asText.map {
    val version = Regex("""name = "rustls-platform-verifier-android"\s+version = "([^"]+)"""").find(it)?.groupValues?.get(1)
        ?: error("rustls-platform-verifier-android is missing from Cargo.lock")
    "org.rustls:rustls-platform-verifier:$version"
}

dependencies {
    api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
    implementation("androidx.annotation:annotation:1.11.0")
    implementation("net.java.dev.jna:jna:5.19.1@aar")
    addProvider("implementation", rustlsPlatformVerifier)

    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
}
