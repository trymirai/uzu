# Native code reaches these classes by name, which R8 can't see: JNA and the UniFFI bindings it marshals,
# and the rustls-platform-verifier classes Rust calls through JNI.
-keep class com.sun.jna.** { *; }
-keep class * extends com.sun.jna.** { *; }
-keep class com.trymirai.uzu.** { *; }
-keep, includedescriptorclasses class org.rustls.platformverifier.** { *; }
-dontwarn java.awt.**
