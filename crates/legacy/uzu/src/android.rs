//! An Android app process lacks what the engine takes from the OS elsewhere: HTTPS certificate checks and audio device
//! queries run through the JVM, and the app picks its own storage root. Kotlin's `Uzu.initialize` hands these over
//! once, before the engine starts.

use std::{
    path::PathBuf,
    sync::{Once, OnceLock},
};

use jni::{
    EnvUnowned,
    errors::{Error, ThrowRuntimeExAndDefault},
    jni_mangle,
    objects::{JClass, JObject, JString},
};

static HOME_PATH: OnceLock<PathBuf> = OnceLock::new();
static ANDROID_CONTEXT: Once = Once::new();

pub fn home_path() -> Option<PathBuf> {
    HOME_PATH.get().cloned()
}

#[jni_mangle("com.trymirai.uzu.Uzu")]
pub fn native_initialize<'caller>(
    mut unowned_env: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    context: JObject<'caller>,
    home_path: JString<'caller>,
) {
    unowned_env
        .with_env(|env| -> Result<(), Error> {
            let home_path = PathBuf::from(home_path.try_to_string(env)?);
            HOME_PATH.get_or_init(|| home_path);
            // ndk-context, which cpal reads to query audio devices, keeps the VM and a global reference to the app
            // context that is never deleted, so the context stays valid for the life of the process.
            if !ANDROID_CONTEXT.is_completed() {
                let java_vm = env.get_java_vm()?.get_raw();
                let context = env.new_global_ref(&context)?.into_raw();
                ANDROID_CONTEXT
                    .call_once(|| unsafe { ndk_context::initialize_android_context(java_vm.cast(), context.cast()) });
            }
            rustls_platform_verifier::android::init_with_env(env, context)
        })
        .resolve::<ThrowRuntimeExAndDefault>()
}
