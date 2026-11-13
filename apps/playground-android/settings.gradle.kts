pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode = RepositoriesMode.FAIL_ON_PROJECT_REPOS
    repositories {
        google()
        mavenCentral()
        // The JVM half of rustls-platform-verifier, which the native library calls to check HTTPS certificates.
        maven("https://github.com/rustls/rustls-platform-verifier/raw/maven-archive/android-release-support/maven/") {
            content {
                includeGroup("org.rustls")
            }
        }
    }
}

rootProject.name = "MiraiPlayground"

include(":app")
include(":uzu")
project(":uzu").projectDir = file("../../crates/legacy/uzu/bindings/kotlin")
