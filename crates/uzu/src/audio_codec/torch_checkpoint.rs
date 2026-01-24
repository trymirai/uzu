use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek},
    path::Path,
    rc::Rc,
};

use thiserror::Error;
use zip::ZipArchive;

use crate::DataType;

#[derive(Debug, Error)]
pub enum TorchCheckpointError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("Pickle error: {0}")]
    Pickle(String),
    #[error("Unsupported torch storage type: {0}")]
    UnsupportedStorageType(String),
    #[error("Tensor \"{name}\" has non-contiguous strides {stride:?} for shape {shape:?} (not yet supported)")]
    NonContiguousTensor {
        name: String,
        shape: Box<[usize]>,
        stride: Box<[usize]>,
    },
    #[error("Tensor \"{name}\" data out of bounds (storage bytes {storage_bytes}, need {need_bytes} at offset {offset_bytes})")]
    TensorOutOfBounds {
        name: String,
        storage_bytes: usize,
        need_bytes: usize,
        offset_bytes: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorchDType {
    F32,
    I32,
}

impl TorchDType {
    pub fn element_size_bytes(self) -> usize {
        match self {
            TorchDType::F32 => 4,
            TorchDType::I32 => 4,
        }
    }

    pub fn to_data_type(self) -> DataType {
        match self {
            TorchDType::F32 => DataType::F32,
            TorchDType::I32 => DataType::I32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TorchTensor {
    pub dtype: TorchDType,
    pub shape: Box<[usize]>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct TorchTensorSpec {
    pub dtype: TorchDType,
    pub shape: Box<[usize]>,
    pub stride: Box<[usize]>,
    pub storage_key: String,
    pub storage_offset_elems: usize,
}

#[derive(Debug, Default)]
pub struct TorchStateDict {
    pub audio_encoder: HashMap<String, TorchTensorSpec>,
    pub audio_decoder: HashMap<String, TorchTensorSpec>,
    pub vector_quantizer: HashMap<String, TorchTensorSpec>,
}

pub struct TorchCheckpoint<R: Read + Seek> {
    archive: ZipArchive<R>,
    pub state_dict: TorchStateDict,
}

impl TorchCheckpoint<File> {
    pub fn open_from_path(
        path: &Path,
    ) -> Result<Self, TorchCheckpointError> {
        let file = File::open(path)?;
        let archive = ZipArchive::new(file)?;
        Self::open_from_archive(archive)
    }
}

impl<R: Read + Seek> TorchCheckpoint<R> {
    pub fn open_from_archive(
        mut archive: ZipArchive<R>,
    ) -> Result<Self, TorchCheckpointError> {
        // Validate byteorder (PyTorch uses a zip member for this).
        if let Ok(mut f) = archive.by_name("model_weights/byteorder") {
            let mut s = String::new();
            f.read_to_string(&mut s)?;
            let s = s.trim();
            if s != "little" {
                return Err(TorchCheckpointError::Pickle(format!(
                    "Unsupported torch byteorder {s:?} (expected \"little\")"
                )));
            }
        }

        let mut pkl = Vec::new();
        archive
            .by_name("model_weights/data.pkl")?
            .read_to_end(&mut pkl)?;

        let state_dict = PickleMachine::new(&pkl).parse_root_state_dict()?;

        Ok(Self {
            archive,
            state_dict,
        })
    }

    pub fn load_tensor(
        &mut self,
        module: TorchModule,
        name: &str,
    ) -> Result<TorchTensor, TorchCheckpointError> {
        let spec = match module {
            TorchModule::AudioEncoder => self.state_dict.audio_encoder.get(name).cloned().ok_or_else(|| {
                TorchCheckpointError::Pickle(format!(
                    "Missing audio_encoder tensor {name}"
                ))
            })?,
            TorchModule::AudioDecoder => self.state_dict.audio_decoder.get(name).cloned().ok_or_else(|| {
                TorchCheckpointError::Pickle(format!(
                    "Missing audio_decoder tensor {name}"
                ))
            })?,
            TorchModule::VectorQuantizer => self.state_dict.vector_quantizer.get(name).cloned().ok_or_else(|| {
                TorchCheckpointError::Pickle(format!(
                    "Missing vector_quantizer tensor {name}"
                ))
            })?,
        };
        self.load_tensor_from_spec(name, &spec)
    }

    pub fn load_tensor_from_spec(
        &mut self,
        name: &str,
        spec: &TorchTensorSpec,
    ) -> Result<TorchTensor, TorchCheckpointError> {
        let path = format!("model_weights/data/{}", spec.storage_key);
        let mut f = self.archive.by_name(&path)?;
        let mut storage = Vec::with_capacity(f.size() as usize);
        f.read_to_end(&mut storage)?;

        let elem_bytes = spec.dtype.element_size_bytes();
        let numel: usize = spec.shape.iter().product();
        let expected_stride = contiguous_stride(&spec.shape);
        if spec.stride.as_ref() != expected_stride.as_slice() {
            return Err(TorchCheckpointError::NonContiguousTensor {
                name: name.to_string(),
                shape: spec.shape.clone(),
                stride: spec.stride.clone(),
            });
        }

        let begin = spec.storage_offset_elems * elem_bytes;
        let need = numel * elem_bytes;
        let end = begin + need;
        if end > storage.len() {
            return Err(TorchCheckpointError::TensorOutOfBounds {
                name: name.to_string(),
                storage_bytes: storage.len(),
                need_bytes: need,
                offset_bytes: begin,
            });
        }

        Ok(TorchTensor {
            dtype: spec.dtype,
            shape: spec.shape.clone(),
            data: storage[begin..end].to_vec(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorchModule {
    AudioEncoder,
    AudioDecoder,
    VectorQuantizer,
}

fn contiguous_stride(shape: &[usize]) -> Vec<usize> {
    let mut stride = vec![0usize; shape.len()];
    let mut s = 1usize;
    for (i, &dim) in shape.iter().enumerate().rev() {
        stride[i] = s;
        s = s.saturating_mul(dim.max(1));
    }
    stride
}

#[derive(Debug, Clone)]
struct GlobalRef {
    module: String,
    name: String,
}

#[derive(Debug, Clone)]
struct StorageRef {
    dtype: TorchDType,
    key: String,
    _numel: usize,
}

#[derive(Clone)]
enum Value {
    Int(i64),
    Bool(bool),
    String(String),
    Tuple(Vec<Value>),
    Dict(Rc<std::cell::RefCell<Vec<(Value, Value)>>>),
    Global(GlobalRef),
    Storage(StorageRef),
    Tensor(TorchTensorSpec),
    RootDict,
}

struct PickleMachine<'a> {
    bytes: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: Vec<Option<Value>>,
    root_created: bool,
    root: TorchStateDict,
}

impl<'a> PickleMachine<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: Vec::new(),
            root_created: false,
            root: TorchStateDict::default(),
        }
    }

    fn parse_root_state_dict(
        mut self,
    ) -> Result<TorchStateDict, TorchCheckpointError> {
        while self.pos < self.bytes.len() {
            let op = self.read_u8()?;
            match op {
                0x80 => {
                    // PROTO
                    let _v = self.read_u8()?;
                },
                b'c' => {
                    // GLOBAL: module\\nname\\n
                    let module = self.read_line()?;
                    let name = self.read_line()?;
                    self.stack.push(Value::Global(GlobalRef { module, name }));
                },
                b'(' => {
                    // MARK
                    self.marks.push(self.stack.len());
                },
                b')' => {
                    // EMPTY_TUPLE
                    self.stack.push(Value::Tuple(Vec::new()));
                },
                b'}' => {
                    // EMPTY_DICT
                    self.stack.push(Value::Dict(Rc::new(std::cell::RefCell::new(Vec::new()))));
                },
                b'X' => {
                    // BINUNICODE (u32 len + bytes)
                    let len = self.read_u32_le()? as usize;
                    let s = self.read_utf8(len)?;
                    self.stack.push(Value::String(s));
                },
                b'K' => {
                    // BININT1
                    let v = self.read_u8()? as i64;
                    self.stack.push(Value::Int(v));
                },
                b'M' => {
                    // BININT2
                    let v = self.read_u16_le()? as i64;
                    self.stack.push(Value::Int(v));
                },
                b'J' => {
                    // BININT (i32)
                    let v = self.read_i32_le()? as i64;
                    self.stack.push(Value::Int(v));
                },
                0x89 => {
                    // NEWFALSE
                    self.stack.push(Value::Bool(false));
                },
                b't' => {
                    // TUPLE (MARK ... items)
                    let mark = self
                        .marks
                        .pop()
                        .ok_or_else(|| TorchCheckpointError::Pickle("TUPLE without MARK".into()))?;
                    let items = self.stack.split_off(mark);
                    self.stack.push(Value::Tuple(items));
                },
                0x85 => {
                    // TUPLE1
                    let v = self.pop()?;
                    self.stack.push(Value::Tuple(vec![v]));
                },
                0x87 => {
                    // TUPLE3
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b, c]));
                },
                b'q' => {
                    // BINPUT
                    let idx = self.read_u8()? as usize;
                    self.memo_set(idx)?;
                },
                b'r' => {
                    // LONG_BINPUT
                    let idx = self.read_u32_le()? as usize;
                    self.memo_set(idx)?;
                },
                b'h' => {
                    // BINGET
                    let idx = self.read_u8()? as usize;
                    let v = self.memo_get(idx)?;
                    self.stack.push(v);
                },
                b'j' => {
                    // LONG_BINGET
                    let idx = self.read_u32_le()? as usize;
                    let v = self.memo_get(idx)?;
                    self.stack.push(v);
                },
                b'Q' => {
                    // BINPERSID
                    let pid = self.pop()?;
                    let storage = self.persistent_load(pid)?;
                    self.stack.push(storage);
                },
                b'R' => {
                    // REDUCE
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let out = self.reduce(callable, args)?;
                    self.stack.push(out);
                },
                b's' => {
                    // SETITEM
                    let value = self.pop()?;
                    let key = self.pop()?;
                    self.dict_set_item(key, value)?;
                },
                b'u' => {
                    // SETITEMS
                    let mark = self
                        .marks
                        .pop()
                        .ok_or_else(|| TorchCheckpointError::Pickle("SETITEMS without MARK".into()))?;
                    let items = self.stack.split_off(mark);
                    if items.len() % 2 != 0 {
                        return Err(TorchCheckpointError::Pickle(
                            "SETITEMS expected even number of items".into(),
                        ));
                    }
                    for pair in items.chunks_exact(2) {
                        let key = pair[0].clone();
                        let value = pair[1].clone();
                        self.dict_set_item(key, value)?;
                    }
                },
                b'b' => {
                    // BUILD: ignore state and keep instance
                    let _state = self.pop()?;
                    let inst = self.pop()?;
                    self.stack.push(inst);
                },
                b'.' => {
                    // STOP
                    return Ok(self.root);
                },
                other => {
                    return Err(TorchCheckpointError::Pickle(format!(
                        "Unsupported pickle opcode 0x{other:02x} at pos {}",
                        self.pos.saturating_sub(1)
                    )));
                },
            }
        }
        Err(TorchCheckpointError::Pickle(
            "Unexpected end of pickle stream".into(),
        ))
    }

    fn dict_set_item(
        &mut self,
        key: Value,
        value: Value,
    ) -> Result<(), TorchCheckpointError> {
        let dict = self
            .stack
            .last()
            .ok_or_else(|| TorchCheckpointError::Pickle("SETITEM with empty stack".into()))?;
        match dict {
            Value::RootDict => {
                let key = match key {
                    Value::String(s) => s,
                    _ => return Ok(()),
                };
                let Value::Tensor(spec) = value else {
                    return Ok(());
                };
                if let Some(stripped) = key.strip_prefix("audio_encoder.") {
                    self.root
                        .audio_encoder
                        .insert(stripped.to_string(), spec);
                } else if let Some(stripped) = key.strip_prefix("audio_decoder.") {
                    self.root
                        .audio_decoder
                        .insert(stripped.to_string(), spec);
                } else if let Some(stripped) = key.strip_prefix("vector_quantizer.") {
                    self.root
                        .vector_quantizer
                        .insert(stripped.to_string(), spec);
                }
                Ok(())
            },
            Value::Dict(m) => {
                m.borrow_mut().push((key, value));
                Ok(())
            },
            _ => Ok(()),
        }
    }

    fn reduce(
        &mut self,
        callable: Value,
        args: Value,
    ) -> Result<Value, TorchCheckpointError> {
        let Value::Global(global) = callable else {
            return Err(TorchCheckpointError::Pickle(
                "REDUCE expected GLOBAL callable".into(),
            ));
        };
        let Value::Tuple(args) = args else {
            return Err(TorchCheckpointError::Pickle(
                "REDUCE expected tuple args".into(),
            ));
        };

        match (global.module.as_str(), global.name.as_str()) {
            ("collections", "OrderedDict") => {
                if !self.root_created {
                    self.root_created = true;
                    Ok(Value::RootDict)
                } else {
                    Ok(Value::Dict(Rc::new(std::cell::RefCell::new(Vec::new()))))
                }
            },
            ("torch._utils", "_rebuild_tensor_v2") => {
                // Args: (storage, storage_offset, size, stride, requires_grad, backward_hooks)
                if args.len() != 6 {
                    return Err(TorchCheckpointError::Pickle(format!(
                        "_rebuild_tensor_v2 expected 6 args, got {}",
                        args.len()
                    )));
                }

                let Value::Storage(storage) = args[0].clone() else {
                    return Err(TorchCheckpointError::Pickle(
                        "_rebuild_tensor_v2 arg0 must be Storage".into(),
                    ));
                };
                let storage_offset_elems = as_usize(&args[1])?;
                let shape = tuple_usizes(&args[2])?;
                let stride = tuple_usizes(&args[3])?;

                Ok(Value::Tensor(TorchTensorSpec {
                    dtype: storage.dtype,
                    shape,
                    stride,
                    storage_key: storage.key,
                    storage_offset_elems,
                }))
            },
            _ => Err(TorchCheckpointError::Pickle(format!(
                "Unsupported REDUCE callable {}.{}",
                global.module, global.name
            ))),
        }
    }

    fn persistent_load(
        &self,
        pid: Value,
    ) -> Result<Value, TorchCheckpointError> {
        let Value::Tuple(items) = pid else {
            return Err(TorchCheckpointError::Pickle(
                "BINPERSID expected tuple pid".into(),
            ));
        };
        if items.len() != 5 {
            return Err(TorchCheckpointError::Pickle(format!(
                "Unsupported persistent id tuple length {}",
                items.len()
            )));
        }
        let Value::String(kind) = &items[0] else {
            return Err(TorchCheckpointError::Pickle(
                "persistent id kind must be string".into(),
            ));
        };
        if kind != "storage" {
            return Err(TorchCheckpointError::Pickle(format!(
                "Unsupported persistent id kind {kind:?}"
            )));
        }
        let Value::Global(storage_type) = &items[1] else {
            return Err(TorchCheckpointError::Pickle(
                "persistent id storage type must be GLOBAL".into(),
            ));
        };
        let dtype = match (storage_type.module.as_str(), storage_type.name.as_str()) {
            ("torch", "FloatStorage") => TorchDType::F32,
            ("torch", "IntStorage") => TorchDType::I32,
            _ => {
                return Err(TorchCheckpointError::UnsupportedStorageType(format!(
                    "{}.{}",
                    storage_type.module, storage_type.name
                )));
            },
        };
        let Value::String(key) = &items[2] else {
            return Err(TorchCheckpointError::Pickle(
                "persistent id storage key must be string".into(),
            ));
        };
        let numel = as_usize(&items[4])?;
        Ok(Value::Storage(StorageRef {
            dtype,
            key: key.clone(),
            _numel: numel,
        }))
    }

    fn memo_set(
        &mut self,
        idx: usize,
    ) -> Result<(), TorchCheckpointError> {
        let v = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| TorchCheckpointError::Pickle("BINPUT with empty stack".into()))?;
        if self.memo.len() <= idx {
            self.memo.resize_with(idx + 1, || None);
        }
        self.memo[idx] = Some(v);
        Ok(())
    }

    fn memo_get(
        &self,
        idx: usize,
    ) -> Result<Value, TorchCheckpointError> {
        self.memo
            .get(idx)
            .and_then(|v| v.clone())
            .ok_or_else(|| TorchCheckpointError::Pickle(format!("Missing memo index {idx}")))
    }

    fn pop(&mut self) -> Result<Value, TorchCheckpointError> {
        self.stack
            .pop()
            .ok_or_else(|| TorchCheckpointError::Pickle("Unexpected empty stack".into()))
    }

    fn read_u8(&mut self) -> Result<u8, TorchCheckpointError> {
        let b = *self
            .bytes
            .get(self.pos)
            .ok_or_else(|| TorchCheckpointError::Pickle("Unexpected EOF".into()))?;
        self.pos += 1;
        Ok(b)
    }

    fn read_u16_le(&mut self) -> Result<u16, TorchCheckpointError> {
        let b0 = self.read_u8()? as u16;
        let b1 = self.read_u8()? as u16;
        Ok(b0 | (b1 << 8))
    }

    fn read_u32_le(&mut self) -> Result<u32, TorchCheckpointError> {
        let b0 = self.read_u8()? as u32;
        let b1 = self.read_u8()? as u32;
        let b2 = self.read_u8()? as u32;
        let b3 = self.read_u8()? as u32;
        Ok(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
    }

    fn read_i32_le(&mut self) -> Result<i32, TorchCheckpointError> {
        Ok(self.read_u32_le()? as i32)
    }

    fn read_line(&mut self) -> Result<String, TorchCheckpointError> {
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.bytes.len() {
            return Err(TorchCheckpointError::Pickle(
                "GLOBAL missing newline".into(),
            ));
        }
        let line = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|e| TorchCheckpointError::Pickle(format!("Invalid UTF-8: {e}")))?;
        self.pos += 1; // consume newline
        Ok(line.to_string())
    }

    fn read_utf8(
        &mut self,
        len: usize,
    ) -> Result<String, TorchCheckpointError> {
        let end = self.pos + len;
        let slice = self.bytes.get(self.pos..end).ok_or_else(|| {
            TorchCheckpointError::Pickle("BINUNICODE out of bounds".into())
        })?;
        self.pos = end;
        std::str::from_utf8(slice)
            .map(|s| s.to_string())
            .map_err(|e| TorchCheckpointError::Pickle(format!("Invalid UTF-8: {e}")))
    }
}

fn as_usize(v: &Value) -> Result<usize, TorchCheckpointError> {
    match v {
        Value::Int(i) => (*i)
            .try_into()
            .map_err(|_| TorchCheckpointError::Pickle("negative int".into())),
        _ => Err(TorchCheckpointError::Pickle(
            "Expected integer".into(),
        )),
    }
}

fn tuple_usizes(v: &Value) -> Result<Box<[usize]>, TorchCheckpointError> {
    let Value::Tuple(items) = v else {
        return Err(TorchCheckpointError::Pickle(
            "Expected tuple".into(),
        ));
    };
    items
        .iter()
        .map(as_usize)
        .collect::<Result<Vec<_>, _>>()
        .map(Vec::into_boxed_slice)
}

