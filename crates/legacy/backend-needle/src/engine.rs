use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use shoji::{
    traits::backend::{State as StateTrait, chat_message::Output},
    types::session::chat::{ChatMessage, ChatReplyConfig},
};

use crate::{
    config::require_needle3,
    error::Error,
    ffi::{Ffi, Lib, complete_into, init_with},
    mapping::{
        bind_tools, complete_inputs, map_envelope, messages::check_unsupported, response::empty_stop_output,
        system_text,
    },
};

static NEXT_STATE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct NeedleState {
    pub id: u64,
    pub fed_up_to: usize,
    pub bound_tools_json: Option<String>,
    pub bound_system: Option<String>,
    pub bound_grammar: Option<String>,
}

impl NeedleState {
    pub fn new() -> Self {
        Self {
            id: NEXT_STATE_ID.fetch_add(1, Ordering::Relaxed),
            fed_up_to: 0,
            bound_tools_json: None,
            bound_system: None,
            bound_grammar: None,
        }
    }
}

impl Default for NeedleState {
    fn default() -> Self {
        Self::new()
    }
}

impl StateTrait for NeedleState {}

struct LoadedWeights {
    path: PathBuf,
    size: usize,
}

struct Inner {
    ffi: Ffi,
    loaded: Option<LoadedWeights>,
    active_owner: Option<u64>,
}

pub struct EngineHandle {
    inner: Arc<Mutex<Inner>>,
}

impl Clone for EngineHandle {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl EngineHandle {
    pub fn new(lib: Lib) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                ffi: lib.ffi,
                loaded: None,
                active_owner: None,
            })),
        }
    }

    pub fn from_ffi(ffi: Ffi) -> Self {
        Self::new(Lib::from_ffi(ffi))
    }

    pub fn load(
        &self,
        path: &Path,
    ) -> Result<usize, Error> {
        require_needle3(path)?;
        let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let data = fs::read(&canonical).map_err(|error| Error::ReadFailed {
            path: canonical.clone(),
            message: error.to_string(),
        })?;
        let mut inner = self.lock();
        if let Some(loaded) = &inner.loaded {
            if loaded.path == canonical {
                return Ok(loaded.size);
            }
            return Err(Error::WeightsLocked {
                loaded: loaded.path.clone(),
                requested: canonical,
            });
        }
        let rc = unsafe { (inner.ffi.load)(data.as_ptr(), data.len() as u64) };
        if rc < 0 {
            return Err(Error::LoadFailed {
                path: canonical,
            });
        }
        let size = data.len();
        inner.loaded = Some(LoadedWeights {
            path: canonical,
            size,
        });
        Ok(size)
    }

    pub fn loaded_size(&self) -> Option<usize> {
        self.lock().loaded.as_ref().map(|loaded| loaded.size)
    }

    pub fn drive_turn(
        &self,
        state: &mut NeedleState,
        input: &[ChatMessage],
        config: &ChatReplyConfig,
    ) -> Result<Output, Error> {
        check_unsupported(input)?;
        let bound = bind_tools(input, config.grammar.as_ref())?;
        let system = system_text(input);

        let mut inner = self.lock();
        let mut just_inited = false;
        if state.bound_tools_json.as_ref() != Some(&bound.fingerprint)
            || state.bound_system.as_ref() != Some(&system)
            || state.bound_grammar != bound.grammar_fingerprint
        {
            init_with(inner.ffi, &system, &bound.json)?;
            state.fed_up_to = 0;
            state.bound_tools_json = Some(bound.fingerprint.clone());
            state.bound_system = Some(system);
            state.bound_grammar = bound.grammar_fingerprint.clone();
            just_inited = true;
        }

        if inner.active_owner != Some(state.id) {
            if !just_inited {
                unsafe {
                    (inner.ffi.reset)();
                }
            }
            inner.active_owner = Some(state.id);
            state.fed_up_to = 0;
        }

        if input.len() < state.fed_up_to {
            unsafe {
                (inner.ffi.reset)();
            }
            state.fed_up_to = 0;
        }

        let increment = &input[state.fed_up_to.min(input.len())..];
        let completes = complete_inputs(increment)?;
        let max_new_tokens = config.token_limit.unwrap_or(512) as i32;
        let started = Instant::now();
        let mut last_raw = None;
        for item in completes {
            last_raw = Some(complete_into(inner.ffi, &item.as_complete_text()?, max_new_tokens)?);
        }
        let duration = started.elapsed().as_secs_f64();
        state.fed_up_to = input.len();

        match last_raw {
            None => Ok(empty_stop_output(duration)),
            Some(raw) => {
                let envelope = serde_json::from_str(&raw).map_err(|error| Error::InvalidEnvelope {
                    message: error.to_string(),
                })?;
                map_envelope(envelope, duration, bound.is_grammar)
            },
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|error| error.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        ffi::{CStr, CString, c_char},
        fs, ptr,
        sync::Mutex,
    };

    use shoji::types::{
        basic::Value,
        session::chat::{ChatContentBlock, ChatMessage, ChatReplyConfig, ChatReplyFinishReason},
    };

    use super::*;
    use crate::{config::NEEDLE3_TAG, ffi::Ffi, mapping::tools::tools_json};

    struct CallLog {
        loads: usize,
        inits: usize,
        completes: Vec<String>,
        resets: usize,
    }

    static LOG: Mutex<CallLog> = Mutex::new(CallLog {
        loads: 0,
        inits: 0,
        completes: Vec::new(),
        resets: 0,
    });

    fn reset_log() {
        let mut log = LOG.lock().unwrap();
        *log = CallLog {
            loads: 0,
            inits: 0,
            completes: Vec::new(),
            resets: 0,
        };
    }

    unsafe extern "C" fn mock_load(
        _data: *const u8,
        _size: u64,
    ) -> i32 {
        LOG.lock().unwrap().loads += 1;
        0
    }

    unsafe extern "C" fn mock_init(
        _system: *const c_char,
        _tools: *const c_char,
        _index: *const c_char,
    ) -> i32 {
        LOG.lock().unwrap().inits += 1;
        0
    }

    unsafe extern "C" fn mock_complete(
        input: *const c_char,
        _max_new_tokens: i32,
        output: *mut c_char,
        output_len: i32,
    ) -> i32 {
        let text = unsafe { CStr::from_ptr(input) }.to_string_lossy().into_owned();
        LOG.lock().unwrap().completes.push(text);
        let json = CString::new(
            r#"{"type":"call","success":true,"function_calls":[{"name":"get_weather","arguments":{"city":"Lagos"}}]}"#,
        )
        .unwrap();
        let bytes = json.as_bytes_with_nul();
        if bytes.len() as i32 > output_len {
            return -1;
        }
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), output as *mut u8, bytes.len());
        }
        0
    }

    unsafe extern "C" fn mock_reset() {
        LOG.lock().unwrap().resets += 1;
    }

    fn mock_handle() -> EngineHandle {
        EngineHandle::from_ffi(Ffi {
            load: mock_load,
            init: mock_init,
            complete: mock_complete,
            reset: mock_reset,
        })
    }

    fn needle3_file() -> tempfile::NamedTempFile {
        let file = tempfile::NamedTempFile::new().unwrap();
        fs::write(file.path(), NEEDLE3_TAG.to_le_bytes()).unwrap();
        file
    }

    #[test]
    fn load_reuses_same_path_and_locks_different_path() {
        reset_log();
        let handle = mock_handle();
        let first = needle3_file();
        let second = needle3_file();
        handle.load(first.path()).unwrap();
        handle.load(first.path()).unwrap();
        assert_eq!(LOG.lock().unwrap().loads, 1);
        let error = handle.load(second.path()).unwrap_err();
        assert!(matches!(error, Error::WeightsLocked { .. }));
    }

    #[test]
    fn drive_turn_increments_then_feeds_tool_array() {
        reset_log();
        let handle = mock_handle();
        let mut state = NeedleState::new();
        let first = vec![ChatMessage::user().with_text("weather in Lagos".to_string())];
        let output = handle.drive_turn(&mut state, &first, &ChatReplyConfig::default()).unwrap();
        assert_eq!(output.finish_reason, Some(ChatReplyFinishReason::ToolCalls));
        assert_eq!(LOG.lock().unwrap().inits, 1);
        assert_eq!(LOG.lock().unwrap().completes, vec!["weather in Lagos".to_string()]);
        assert_eq!(state.fed_up_to, 1);

        let second = vec![
            ChatMessage::user().with_text("weather in Lagos".to_string()),
            ChatMessage::assistant().with_text(String::new()),
            ChatMessage::tool().with_block(ChatContentBlock::ToolCallResult {
                identifier: Some("needle-0".to_string()),
                name: Some("get_weather".to_string()),
                value: Value {
                    json: r#"{"city":"Lagos"}"#.to_string(),
                },
            }),
        ];
        handle.drive_turn(&mut state, &second, &ChatReplyConfig::default()).unwrap();
        let log = LOG.lock().unwrap();
        assert_eq!(log.inits, 1);
        assert_eq!(log.completes.len(), 2);
        assert_eq!(log.completes[1], r#"[{"city":"Lagos"}]"#);
        assert_eq!(log.resets, 0);
        assert_eq!(state.fed_up_to, 3);
    }

    #[test]
    fn owner_switch_resets_and_replays() {
        reset_log();
        let handle = mock_handle();
        let mut first = NeedleState::new();
        let mut second = NeedleState::new();
        let messages = vec![ChatMessage::user().with_text("one".to_string())];
        handle.drive_turn(&mut first, &messages, &ChatReplyConfig::default()).unwrap();
        handle.drive_turn(&mut second, &messages, &ChatReplyConfig::default()).unwrap();
        handle.drive_turn(&mut first, &messages, &ChatReplyConfig::default()).unwrap();
        let log = LOG.lock().unwrap();
        assert_eq!(log.inits, 2);
        assert!(log.resets >= 1);
        assert_eq!(log.completes, vec!["one".to_string(), "one".to_string(), "one".to_string()]);
    }

    #[test]
    fn tools_change_reinitializes() {
        reset_log();
        let handle = mock_handle();
        let mut state = NeedleState::new();
        let first = vec![ChatMessage::user().with_text("one".to_string())];
        handle.drive_turn(&mut state, &first, &ChatReplyConfig::default()).unwrap();

        let tools = ChatMessage::developer().with_tool_namespaces(vec![shoji::types::basic::ToolNamespace {
            name: "default".to_string(),
            description: None,
            tools: vec![shoji::types::basic::ToolDescription::Function {
                tool_function: shoji::types::basic::ToolFunction {
                    name: "get_weather".to_string(),
                    description: "weather".to_string(),
                    parameters: None,
                    return_definition: None,
                },
            }],
        }]);
        let second = vec![tools, ChatMessage::user().with_text("one".to_string())];
        handle.drive_turn(&mut state, &second, &ChatReplyConfig::default()).unwrap();
        assert_eq!(LOG.lock().unwrap().inits, 2);
        let json = tools_json(&second).unwrap();
        assert!(json.contains("get_weather"));
    }
}
