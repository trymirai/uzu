plugins {
    kotlin("multiplatform") version "2.4.21"
}

group = "com.trymirai"

kotlin {
    jvm()

    sourceSets {
        commonMain {
            // The UniFFI bindings and their extensions, written by `cargo tools build kotlin`.
            kotlin.srcDir("src/commonMain/generated")
            dependencies {
                api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
                implementation("net.java.dev.jna:jna:5.19.1")
            }
        }
        jvmMain {
            // The engine library `cargo tools build kotlin` built, in the classpath directory JNA loads it from.
            resources.srcDir("src/jvmMain/natives")
        }
        jvmTest {
            dependencies {
                implementation(kotlin("test"))
                // The structured output example parses its reply with org.json.
                implementation("org.json:json:20260814")
            }
        }
    }
}

tasks.withType<Test>().configureEach {
    testLogging {
        showStandardStreams = true
    }
}

// The examples download models and call cloud APIs, so `check` leaves them out; `cargo tools example` runs one.
val examples = "com/trymirai/uzu/examples/**"

tasks.named<Test>("jvmTest") {
    exclude(examples)
}

tasks.register<Test>("jvmExamples") {
    val testTask = tasks.named<Test>("jvmTest").get()
    testClassesDirs = testTask.testClassesDirs
    classpath = testTask.classpath
    include(examples)
}
