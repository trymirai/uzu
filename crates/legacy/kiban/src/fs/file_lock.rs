use std::{io, path::Path};

pub struct FileLock {
    #[cfg(not(target_family = "wasm"))]
    file: std::fs::File,
    #[cfg(target_family = "wasm")]
    path: std::path::PathBuf,
    #[cfg(target_family = "wasm")]
    release: web_sys::js_sys::Function,
}

impl FileLock {
    pub async fn try_acquire(path: &Path) -> Result<Option<Self>, io::Error> {
        if let Some(parent) = path.parent() {
            super::asyn::create_dir_all(parent).await?;
        }

        #[cfg(target_family = "wasm")]
        {
            use std::io::ErrorKind;

            use futures_util::future::{Either, select};
            use wasm_bindgen_futures::JsFuture;
            use web_sys::{
                js_sys::{Function, Object, Promise, Reflect},
                wasm_bindgen::{JsCast, JsValue, closure::Closure},
            };

            use super::asyn_opfs::js_value_to_io_error;

            let to_io = |value: JsValue| js_value_to_io_error(&value, ErrorKind::Other);
            let navigator =
                web_sys::window().ok_or_else(|| io::Error::other("no window object available"))?.navigator();
            let locks = Reflect::get(&navigator, &JsValue::from_str("locks")).map_err(to_io)?;
            let request = Reflect::get(&locks, &JsValue::from_str("request"))
                .map_err(to_io)?
                .dyn_into::<Function>()
                .map_err(to_io)?;
            let options = Object::new();
            Reflect::set(&options, &JsValue::from_str("ifAvailable"), &JsValue::TRUE).map_err(to_io)?;
            let (granted_sender, granted_receiver) = tokio::sync::oneshot::channel();
            let callback = Closure::once_into_js(move |lock: JsValue| -> JsValue {
                if lock.is_null() {
                    let _ = granted_sender.send(None);
                    return JsValue::UNDEFINED;
                }
                let mut release = None;
                let held = Promise::new(&mut |resolve, _reject| release = Some(resolve));
                let _ = granted_sender.send(release);
                held.into()
            });
            let requested = request
                .call3(&locks, &JsValue::from_str(&path.to_string_lossy()), &options, &callback)
                .map_err(to_io)?;
            let release = match select(granted_receiver, JsFuture::from(Promise::from(requested))).await {
                Either::Left((Ok(release), _)) => release,
                Either::Left((Err(_), _)) | Either::Right((Ok(_), _)) => None,
                Either::Right((Err(error), _)) => return Err(to_io(error)),
            };
            Ok(release.map(|release| Self {
                path: path.to_path_buf(),
                release,
            }))
        }

        #[cfg(not(target_family = "wasm"))]
        {
            let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?;
            match file.try_lock() {
                Ok(()) => Ok(Some(Self {
                    file,
                })),
                Err(std::fs::TryLockError::WouldBlock) => Ok(None),
                Err(std::fs::TryLockError::Error(error)) => Err(error),
            }
        }
    }

    pub async fn write(
        &self,
        contents: &[u8],
    ) -> Result<(), io::Error> {
        #[cfg(target_family = "wasm")]
        {
            super::asyn::write(&self.path, contents).await
        }

        #[cfg(not(target_family = "wasm"))]
        {
            use std::io::{Seek, SeekFrom, Write};

            self.file.set_len(0)?;
            (&self.file).seek(SeekFrom::Start(0))?;
            (&self.file).write_all(contents)
        }
    }
}

#[cfg(target_family = "wasm")]
impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = self.release.call0(&web_sys::wasm_bindgen::JsValue::NULL);
    }
}
