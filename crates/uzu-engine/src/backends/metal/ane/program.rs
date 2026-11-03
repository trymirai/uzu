//! A compiled, loaded multi-function ANE program, and its evaluation.

use std::{
    ffi::CStr,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use block2::RcBlock;
use metal::MTLSharedEvent;
use objc2::{
    msg_send,
    rc::Retained,
    runtime::{AnyObject, Bool, ProtocolObject},
    sel,
};
use objc2_core_foundation::CFRetained;
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};
use objc2_io_surface::IOSurfaceRef;
use parking_lot::Mutex;

use super::{
    AneError, AneSurface,
    objc::{array_items, call_error, dictionary_keys, guarded, number, object_for_key, string, unsigned},
    runtime::{ANE_QOS, AneRuntime},
};

/// A function of the program: its input and output symbols, by name and by index among all the
/// program's symbols (a request takes the indices).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AneProcedure {
    pub function: String,
    pub inputs: Vec<String>,
    pub input_symbols: Vec<u32>,
    pub output_symbol: u32,
}

pub struct AneProgram {
    runtime: Arc<AneRuntime>,
    model: Retained<AnyObject>,
    procedures: Vec<AneProcedure>,
    was_cached: bool,
}

// SAFETY: the `_ANEModel` is only messaged through the framework's thread-safe client.
unsafe impl Send for AneProgram {}
unsafe impl Sync for AneProgram {}

/// A request's arguments without its events: a procedure bound to its surfaces.
pub struct AneBinding {
    procedure: Retained<NSNumber>,
    inputs: Retained<NSArray<AnyObject>>,
    input_indices: Retained<NSArray<NSNumber>>,
    outputs: Retained<NSArray<AnyObject>>,
    output_indices: Retained<NSArray<NSNumber>>,
    _surfaces: Vec<CFRetained<IOSurfaceRef>>,
}

// SAFETY: immutable Foundation collections and retained IOSurfaces.
unsafe impl Send for AneBinding {}
unsafe impl Sync for AneBinding {}

fn fnv1a(
    data: &[u8],
    mut hash: u64,
) -> u64 {
    for byte in data {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn cache_root() -> PathBuf {
    let mut path = vec![0u8; 1024];
    let length = unsafe { libc::confstr(libc::_CS_DARWIN_USER_CACHE_DIR, path.as_mut_ptr().cast(), path.len()) };
    if length > 0
        && length <= path.len()
        && let Ok(directory) = CStr::from_bytes_until_nul(&path)
    {
        return PathBuf::from(directory.to_string_lossy().into_owned()).join("uzu-ane");
    }
    std::env::temp_dir().join("uzu-ane")
}

/// Writes `contents` to `path` unless it already holds them, through a renamed partial file so a
/// concurrent reader never sees a torn file.
fn write_atomically(
    path: &Path,
    contents: &[u8],
) -> Result<(), AneError> {
    if fs::read(path).is_ok_and(|existing| existing == contents) {
        return Ok(());
    }
    let partial = path.with_extension(format!("partial.{}", std::process::id()));
    fs::write(&partial, contents)?;
    fs::rename(&partial, path)?;
    Ok(())
}

fn ns_string(value: &str) -> Retained<NSString> {
    NSString::from_str(value)
}

fn empty_options() -> Retained<NSDictionary<NSString, AnyObject>> {
    NSDictionary::new()
}

fn as_object<T: ?Sized>(value: &T) -> &AnyObject {
    unsafe { &*(value as *const T).cast::<AnyObject>() }
}

impl AneProgram {
    /// Compiles (unless the service holds a compilation of the same source) and loads a program.
    pub fn new(
        runtime: Arc<AneRuntime>,
        mil: &str,
        weights: &[u8],
    ) -> Result<Self, AneError> {
        let key = format!("{:016x}", fnv1a(weights, fnv1a(mil.as_bytes(), 0xcbf2_9ce4_8422_2325)));
        let directory = cache_root().join(&key);
        fs::create_dir_all(&directory)?;
        write_atomically(&directory.join("model.mil"), mil.as_bytes())?;
        write_atomically(&directory.join("weights.bin"), weights)?;

        let model = guarded("model creation", || {
            let path = ns_string(&directory.to_string_lossy());
            let url = NSURL::fileURLWithPath_isDirectory(&path, true);
            let key = ns_string(&key);
            let model: Option<Retained<AnyObject>> =
                unsafe { msg_send![runtime.model_class, modelAtURL: &*url, key: &*key] };
            model.ok_or_else(|| call_error("model creation", None))
        })?;

        let mut program = Self {
            runtime,
            model,
            procedures: Vec::new(),
            was_cached: false,
        };

        program.was_cached = program.compiled()?;
        if !program.was_cached {
            program.compile()?;
        }
        program.load(program.was_cached)?;

        program.procedures = program.describe()?;
        Ok(program)
    }

    fn client(&self) -> &AnyObject {
        &self.runtime.client
    }

    fn compiled(&self) -> Result<bool, AneError> {
        guarded("compiled model lookup", || {
            let exists: bool = unsafe { msg_send![self.client(), compiledModelExistsFor: &*self.model] };
            Ok(exists)
        })
    }

    fn compile(&self) -> Result<(), AneError> {
        guarded("compilation", || {
            let keys = [&*ns_string("kANEFModelType"), &*ns_string("kANEFNetPlistFilenameKey")];
            let values = [&*ns_string("kANEFModelMIL"), &*ns_string("model.mil")];
            let options = NSDictionary::from_slices(&keys, &values);
            let result: Result<(), Retained<NSError>> = unsafe {
                msg_send![self.client(), compileModel: &*self.model, options: &*options, qos: ANE_QOS, error: _]
            };
            result.map_err(|error| call_error("compilation", Some(&error)))
        })
    }

    /// Loads the compiled model. A cached compilation that fails to load (from another compiler,
    /// say) is purged, compiled again and loaded once more.
    fn load(
        &self,
        recompile: bool,
    ) -> Result<(), AneError> {
        let load = || {
            guarded("load", || {
                let options = empty_options();
                let result: Result<(), Retained<NSError>> = unsafe {
                    msg_send![self.client(), loadModel: &*self.model, options: &*options, qos: ANE_QOS, error: _]
                };
                result.map_err(|error| call_error("load", Some(&error)))
            })
        };
        if let Err(error) = load() {
            if !recompile {
                return Err(error);
            }
            guarded("purge", || {
                let _: () = unsafe { msg_send![self.client(), purgeCompiledModel: &*self.model] };
                Ok(())
            })?;
            self.compile()?;
            load()?;
        }
        Ok(())
    }

    fn describe(&self) -> Result<Vec<AneProcedure>, AneError> {
        guarded("model attributes", || {
            let malformed = |what: &str| AneError::Description(what.to_string());
            let attributes: Option<Retained<AnyObject>> = unsafe { msg_send![&*self.model, modelAttributes] };
            let attributes = attributes.ok_or_else(|| malformed("no attributes"))?;
            let description =
                object_for_key(&attributes, "ANEFModelDescription").ok_or_else(|| malformed("no description"))?;
            let symbols = object_for_key(&description, "kANEFModelInputSymbolsArrayKey")
                .and_then(|symbols| array_items(&symbols))
                .ok_or_else(|| malformed("no input symbols"))?;
            let functions = object_for_key(&description, "kANEFModelProcedureNameToIDMapKey")
                .ok_or_else(|| malformed("no procedure names"))?;
            let entries = object_for_key(&description, "ANEFModelProcedures")
                .and_then(|entries| array_items(&entries))
                .ok_or_else(|| malformed("no procedures"))?;

            let mut procedures = vec![AneProcedure::default(); entries.len()];
            for function in dictionary_keys(&functions).ok_or_else(|| malformed("procedure names"))? {
                let name = string(&function).ok_or_else(|| malformed("procedure name"))?;
                let index = object_for_key(&functions, &name)
                    .and_then(|index| unsigned(&index))
                    .ok_or_else(|| malformed("procedure index"))? as usize;
                let procedure = procedures.get_mut(index).ok_or_else(|| malformed("procedure index range"))?;
                procedure.function = name;
            }
            for entry in entries {
                let index = object_for_key(&entry, "ANEFModelProcedureID")
                    .and_then(|index| unsigned(&index))
                    .ok_or_else(|| malformed("procedure id"))? as usize;
                let outputs = object_for_key(&entry, "ANEFModelOutputSymbolIndexArray")
                    .and_then(|outputs| array_items(&outputs))
                    .ok_or_else(|| malformed("output symbols"))?;
                let inputs = object_for_key(&entry, "ANEFModelInputSymbolIndexArray")
                    .and_then(|inputs| array_items(&inputs))
                    .ok_or_else(|| malformed("input symbols"))?;
                if outputs.len() != 1 {
                    return Err(malformed("a procedure of other than one output"));
                }
                let procedure = procedures.get_mut(index).ok_or_else(|| malformed("procedure id range"))?;
                procedure.output_symbol = unsigned(&outputs[0]).ok_or_else(|| malformed("output symbol"))? as u32;
                for symbol in inputs {
                    let symbol = unsigned(&symbol).ok_or_else(|| malformed("input symbol"))? as usize;
                    let name =
                        symbols.get(symbol).and_then(|name| string(name)).ok_or_else(|| malformed("input name"))?;
                    procedure.input_symbols.push(symbol as u32);
                    procedure.inputs.push(name);
                }
            }
            Ok(procedures)
        })
    }

    #[cfg(test)]
    pub fn procedures(&self) -> &[AneProcedure] {
        &self.procedures
    }

    pub fn procedure(
        &self,
        function: &str,
    ) -> Option<usize> {
        self.procedures.iter().position(|procedure| procedure.function == function)
    }

    /// Whether the service already held a compilation of this program's source.
    #[cfg(test)]
    pub fn was_cached(&self) -> bool {
        self.was_cached
    }

    /// Binds a procedure's inputs, by name, and its output to surfaces.
    pub fn bind(
        &self,
        procedure: usize,
        inputs: &[(&str, &AneSurface)],
        output: &AneSurface,
    ) -> Result<AneBinding, AneError> {
        let described =
            self.procedures.get(procedure).ok_or_else(|| AneError::UnknownProcedure(procedure.to_string()))?;
        guarded("binding", || {
            let surface_object = |surface: &AneSurface| -> Result<Retained<AnyObject>, AneError> {
                let object: Option<Retained<AnyObject>> =
                    unsafe { msg_send![self.runtime.surface_class, objectWithIOSurface: surface.io_surface()] };
                object.ok_or_else(|| call_error("surface object creation", None))
            };
            let mut objects = Vec::with_capacity(described.inputs.len());
            let mut indices = Vec::with_capacity(described.inputs.len());
            let mut surfaces = Vec::with_capacity(described.inputs.len() + 1);
            for (name, symbol) in described.inputs.iter().zip(&described.input_symbols) {
                let (_, surface) = inputs
                    .iter()
                    .find(|(input, _)| *input == name.as_str())
                    .ok_or_else(|| AneError::MissingInput(name.clone()))?;
                objects.push(surface_object(surface)?);
                indices.push(number(u64::from(*symbol)));
                surfaces.push(CFRetained::from(surface.io_surface()));
            }
            surfaces.push(CFRetained::from(output.io_surface()));
            Ok(AneBinding {
                procedure: number(procedure as u64),
                inputs: NSArray::from_retained_slice(&objects),
                input_indices: NSArray::from_retained_slice(&indices),
                outputs: NSArray::from_retained_slice(&[surface_object(output)?]),
                output_indices: NSArray::from_retained_slice(&[number(u64::from(described.output_symbol))]),
                _surfaces: surfaces,
            })
        })
    }

    fn request(
        &self,
        binding: &AneBinding,
        events: Option<&AnyObject>,
    ) -> Result<Retained<AnyObject>, AneError> {
        let none: Option<&AnyObject> = None;
        let request: Option<Retained<AnyObject>> = unsafe {
            msg_send![
                self.runtime.request_class,
                requestWithInputs: &*binding.inputs,
                inputIndices: &*binding.input_indices,
                outputs: &*binding.outputs,
                outputIndices: &*binding.output_indices,
                weightsBuffer: none,
                perfStats: none,
                procedureIndex: &*binding.procedure,
                sharedEvents: events,
                transactionHandle: none
            ]
        };
        request.ok_or_else(|| call_error("request creation", None))
    }

    fn evaluate_request(
        &self,
        request: &AnyObject,
    ) -> Result<(), AneError> {
        let options = empty_options();
        let result: Result<(), Retained<NSError>> = unsafe {
            msg_send![self.client(), evaluateWithModel: &*self.model, options: &*options, request: request, qos: ANE_QOS, error: _]
        };
        result.map_err(|error| call_error("evaluation", Some(&error)))
    }

    /// Evaluates a binding synchronously.
    #[cfg(test)]
    pub fn evaluate(
        &self,
        binding: &AneBinding,
    ) -> Result<(), AneError> {
        guarded("evaluation", || {
            let request = self.request(binding, None)?;
            self.evaluate_request(&request)
        })
    }

    /// Queues an evaluation that starts once `event` reaches `wait` and sets `event` to `signal`
    /// when done; returns at once. `done` receives whether the evaluation succeeded.
    pub fn enqueue(
        &self,
        binding: &AneBinding,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        wait: u64,
        signal: u64,
        done: Box<dyn FnOnce(bool) + Send>,
    ) -> Result<(), AneError> {
        let event = as_object(event);
        let shares_port: bool = unsafe { msg_send![event, respondsToSelector: sel!(eventPort)] };
        if !shares_port {
            return Err(AneError::Unavailable(
                "the Metal shared event has no eventPort (Metal validation layer?)".to_string(),
            ));
        }
        guarded("enqueue", || {
            let signal_event: Option<Retained<AnyObject>> = unsafe {
                msg_send![self.runtime.signal_event_class, signalEventWithValue: signal, symbolIndex: 0u32, eventType: 0i64, sharedEvent: event]
            };
            let wait_event: Option<Retained<AnyObject>> = unsafe {
                msg_send![self.runtime.wait_event_class, waitEventWithValue: wait, sharedEvent: event, eventType: 0u64]
            };
            let (Some(signal_event), Some(wait_event)) = (signal_event, wait_event) else {
                return Err(call_error("event creation", None));
            };
            let signals = NSArray::from_retained_slice(&[signal_event]);
            let waits = NSArray::from_retained_slice(&[wait_event]);
            let events: Option<Retained<AnyObject>> = unsafe {
                msg_send![self.runtime.shared_events_class, sharedEventsWithSignalEvents: &*signals, waitEvents: &*waits]
            };
            let events = events.ok_or_else(|| call_error("event creation", None))?;
            let request = self.request(binding, Some(&events))?;

            // An evaluation with shared events runs asynchronously and requires a completion
            // handler, which only reports whether it succeeded.
            let done = Mutex::new(Some(done));
            let handler = RcBlock::new(move |success: Bool, _error: *mut NSError| {
                if let Some(done) = done.lock().take() {
                    done(success.as_bool());
                }
            });
            let _: () = unsafe { msg_send![&*request, setCompletionHandler: &*handler] };
            self.evaluate_request(&request)
        })
    }
}

impl Drop for AneProgram {
    fn drop(&mut self) {
        let _ = guarded("unload", || {
            let options = empty_options();
            let result: Result<(), Retained<NSError>> = unsafe {
                msg_send![self.client(), unloadModel: &*self.model, options: &*options, qos: ANE_QOS, error: _]
            };
            result.map_err(|error| call_error("unload", Some(&error)))
        });
    }
}
