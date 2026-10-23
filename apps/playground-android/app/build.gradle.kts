plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

val workspaceVersion = providers.fileContents(rootProject.layout.projectDirectory.file("../../Cargo.toml")).asText.map {
    Regex("""\[workspace\.package][^\[]*?\nversion = "([^"]+)"""").find(it)?.groupValues?.get(1)
        ?: error("The workspace version is missing from Cargo.toml")
}

android {
    namespace = "com.trymirai.playground"
    compileSdk = libs.versions.compile.sdk.get().toInt()
    ndkVersion = libs.versions.ndk.get()

    defaultConfig {
        applicationId = "com.mirai.tech.playground"
        minSdk = libs.versions.min.sdk.get().toInt()
        targetSdk = libs.versions.compile.sdk.get().toInt()
        versionCode = 1
        versionName = workspaceVersion.get()
        // The engine is built for arm64 alone (the Kotlin targets in .config/platforms.toml).
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
        }
    }

    buildFeatures {
        compose = true
    }

    androidResources {
        // The podcast comes straight from the iOS playground's asset catalog; its catalog metadata stays behind.
        ignoreAssetsPatterns += "!Contents.json"
    }
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addStaticSourceDirectory("../../playground/Resources/Assets.xcassets/AboutUsPodcast.dataset")
    }
}

dependencies {
    implementation(project(":uzu"))
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.viewmodel.navigation3)
    implementation(libs.androidx.navigation3.runtime)
    implementation(libs.androidx.navigation3.ui)
    implementation(libs.kotlinx.serialization.core)
}
