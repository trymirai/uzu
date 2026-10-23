//! An Android app process lacks what the engine takes from the OS elsewhere: HTTPS certificate checks run through the
//! JVM, and the app picks its own storage root. Kotlin's `Uzu.initialize` hands both over once, before the engine starts.

use std::{path::PathBuf, sync::OnceLock};

use jni::{
    EnvUnowned,
    errors::{Error, ThrowRuntimeExAndDefault},
    jni_mangle,
    objects::{JClass, JObject, JString},
};

static HOME_PATH: OnceLock<PathBuf> = OnceLock::new();

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
            rustls_platform_verifier::android::init_with_env(env, context)
        })
        .resolve::<ThrowRuntimeExAndDefault>()
}
