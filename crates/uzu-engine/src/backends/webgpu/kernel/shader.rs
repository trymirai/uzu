use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, HashMap},
    ops::Deref,
    sync::Arc,
};

use super::super::command_buffer::{Binding, WebGpuCommandBufferEncoding};
use crate::data_type::DataType;

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum ParamWords {
    Scalar([u32; 1]),
    Array(Vec<u32>),
}

impl Deref for ParamWords {
    type Target = [u32];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Scalar(value) => value,
            Self::Array(values) => values,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) struct Param {
    // Scalar uniforms dominate decode. Store them inline rather than making a
    // separate heap allocation for every parameter of every dispatch.
    pub words: ParamWords,
    kind: &'static str,
    pub data_type: Option<DataType>,
}
impl Param {
    pub fn u32(v: u32) -> Self {
        Self {
            words: ParamWords::Scalar([v]),
            kind: "u32",
            data_type: None,
        }
    }
    pub fn i32(v: i32) -> Self {
        Self {
            words: ParamWords::Scalar([v as u32]),
            kind: "i32",
            data_type: None,
        }
    }
    pub fn f32(v: f32) -> Self {
        Self {
            words: ParamWords::Scalar([v.to_bits()]),
            kind: "f32",
            data_type: None,
        }
    }
    pub fn bool(v: bool) -> Self {
        Self {
            words: ParamWords::Scalar([u32::from(v)]),
            kind: "bool",
            data_type: None,
        }
    }
    pub fn dtype(v: DataType) -> Self {
        Self {
            words: ParamWords::Scalar([v as u32]),
            kind: "u32",
            data_type: Some(v),
        }
    }
    pub fn words(v: Vec<u32>) -> Self {
        Self {
            words: ParamWords::Array(v),
            kind: "u32",
            data_type: None,
        }
    }
}

#[derive(Clone)]
pub(super) struct NamedBinding {
    name: &'static str,
    binding: Binding,
    dtype: DataType,
}
impl NamedBinding {
    pub fn new(
        name: &'static str,
        binding: Binding,
        dtype: DataType,
    ) -> Self {
        Self {
            name,
            binding,
            dtype,
        }
    }
}

#[derive(Clone)]
pub(super) struct Shader {
    bindings: Vec<NamedBinding>,
    pub specs: BTreeMap<&'static str, Param>,
    pub params: BTreeMap<&'static str, Param>,
    aliases: Vec<(&'static str, &'static str)>,
    source: &'static str,
    pub groups: [u32; 3],
}

/// A shader depends on its layout and specializations, not the current tensor
/// handles, byte offsets or uniform values. Retain that generated source across
/// decode steps instead of rebuilding all accessors and substitutions per token.
#[derive(PartialEq, Eq, Hash)]
struct ShaderKey {
    aliases: Vec<(&'static str, &'static str)>,
    source: (usize, usize),
    specs: BTreeMap<&'static str, Param>,
    params: Vec<(&'static str, &'static str, usize)>,
    bindings: Vec<(&'static str, usize, DataType, bool)>,
}

thread_local! {
    // Bound host memory when a worker loads many different model configurations.
    static SHADER_SOURCES: RefCell<HashMap<ShaderKey, Arc<str>>> = RefCell::new(HashMap::new());
}
impl Shader {
    pub fn new(
        bindings: Vec<NamedBinding>,
        specs: BTreeMap<&'static str, Param>,
        params: Vec<(&'static str, Param)>,
    ) -> Self {
        Self {
            bindings,
            specs,
            params: params.into_iter().collect(),
            aliases: Vec::new(),
            source: "",
            groups: [1, 1, 1],
        }
    }
    pub fn u(
        &self,
        key: &str,
    ) -> u32 {
        self.params.get(key).or_else(|| self.specs.get(key)).map_or(0, |p| p.words[0])
    }
    pub fn bind(
        &mut self,
        binding: NamedBinding,
    ) {
        self.bindings.push(binding);
    }
    pub fn flag(
        &self,
        key: &str,
    ) -> bool {
        self.u(key) != 0
    }
    pub fn set(
        &mut self,
        key: &'static str,
        value: u32,
    ) {
        self.params.insert(key, Param::u32(value));
    }
    pub fn kernel(
        &mut self,
        name: &str,
        width: u32,
    ) {
        self.source = super::compiled_shader(name, width);
    }
    pub fn alias(
        &mut self,
        from: &'static str,
        to: &'static str,
    ) {
        self.aliases.push((from, to));
    }
    pub fn dispatch(
        self,
        command: &mut WebGpuCommandBufferEncoding,
    ) {
        let (source, bindings, uniforms, groups) = self.build();
        if groups.iter().all(|&n| n > 0) {
            command.dispatch_cached(&source, bindings, &uniforms, groups);
        }
    }
    fn build(mut self) -> (Arc<str>, Vec<Binding>, Vec<u32>, [u32; 3]) {
        let mut physical = Vec::<Binding>::new();
        let mut binding_shapes = Vec::with_capacity(self.bindings.len());
        for named in &self.bindings {
            let index = if let Some(index) = physical.iter().position(|binding| binding.id() == named.binding.id()) {
                if named.binding.is_writable() {
                    physical[index] = named.binding.clone();
                }
                index
            } else {
                physical.push(named.binding.clone());
                physical.len() - 1
            };
            binding_shapes.push((named.name, index, named.dtype, named.binding.is_writable()));
        }
        let key = ShaderKey {
            aliases: self.aliases.clone(),
            source: (self.source.as_ptr() as usize, self.source.len()),
            specs: std::mem::take(&mut self.specs),
            params: self.params.iter().map(|(name, value)| (*name, value.kind, value.words.len())).collect(),
            bindings: binding_shapes,
        };
        if let Some(source) = SHADER_SOURCES.with(|cache| cache.borrow().get(&key).cloned()) {
            let uniforms = self
                .params
                .values()
                .flat_map(|value| value.words.iter().copied())
                .chain(self.bindings.iter().map(|named| named.binding.offset))
                .collect();
            return (source, physical, uniforms, self.groups);
        }
        self.specs = key.specs.clone();
        let (source, bindings, uniforms, groups) = self.build_uncached();
        let source: Arc<str> = source.into();
        SHADER_SOURCES.with(|cache| {
            let mut cache = cache.borrow_mut();
            if cache.len() >= 1024 {
                cache.clear();
            }
            cache.insert(key, source.clone());
        });
        (source, bindings, uniforms, groups)
    }

    fn build_uncached(self) -> (String, Vec<Binding>, Vec<u32>, [u32; 3]) {
        let mut uniforms = Vec::<u32>::new();
        let mut substitutions: BTreeMap<String, String> = BTreeMap::new();
        let mut array_offsets = BTreeMap::new();
        for (key, value) in &self.specs {
            let raw = value.words[0];
            let literal = match value.kind {
                "bool" => (raw != 0).to_string(),
                "f32" => format!("bitcast<f32>({raw}u)"),
                "i32" => format!("bitcast<i32>({raw}u)"),
                _ => format!("{raw}u"),
            };
            substitutions.insert((*key).to_owned(), literal);
        }
        for (key, value) in &self.params {
            array_offsets.insert(*key, uniforms.len());
            for (i, word) in value.words.iter().enumerate() {
                let index = uniforms.len();
                uniforms.push(*word);
                let raw = uniform(index);
                let expression = match value.kind {
                    "bool" => format!("({raw} != 0u)"),
                    "f32" => format!("bitcast<f32>({raw})"),
                    "i32" => format!("bitcast<i32>({raw})"),
                    _ => raw,
                };
                substitutions.insert(
                    if i == 0 {
                        (*key).to_owned()
                    } else {
                        format!("{key}_{i}")
                    },
                    expression,
                );
            }
        }
        let mut physical = Vec::<Binding>::new();
        let mut logical = Vec::new();
        for named in self.bindings {
            let idx = if let Some(index) = physical.iter().position(|b| b.id() == named.binding.id()) {
                if named.binding.is_writable() && !physical[index].is_writable() {
                    physical[index] = named.binding.clone();
                }
                index
            } else {
                physical.push(named.binding.clone());
                physical.len() - 1
            };
            let offset_index = uniforms.len();
            uniforms.push(named.binding.offset);
            logical.push((named.name, idx, named.dtype, uniform(offset_index)));
        }
        let mut source = String::new();
        let mut body = String::new();
        // Directives must precede all WGSL declarations. Slang emits feature
        // directives; the build compiler adds its required subgroup-id feature.
        for line in self.source.lines() {
            if line.starts_with("enable ") || line.starts_with("requires ") {
                source.push_str(line);
                source.push('\n');
            } else {
                body.push_str(line);
                body.push('\n');
            }
        }
        for (from, to) in self.aliases {
            body = body
                .replace(&format!("load_{from}("), &format!("load_{to}("))
                .replace(&format!("word_{from}("), &format!("word_{to}("));
        }
        for (i, b) in physical.iter().enumerate() {
            let (access, ty) = if b.is_writable() {
                ("read_write", "atomic<u32>")
            } else {
                ("read", "u32")
            };
            source.push_str(&format!("@group(0) @binding({i}) var<storage, {access}> b{i}: array<{ty}>;\n"));
        }
        let count = uniforms.len().max(1).div_ceil(4);
        source.push_str(&format!(
            "struct Params {{ data: array<vec4<u32>, {count}> }}\n@group(0) @binding({}) var<uniform> p: Params;\n",
            physical.len()
        ));
        source.push_str(STORAGE_FORMATS);
        // These typed functions are the link boundary declared in abi.slang.
        // Specializations are literals; per-dispatch values remain uniforms.
        for line in self.source.lines() {
            let (declaration, array) = if let Some(declaration) = line.strip_prefix("// uzu-param: ") {
                (declaration, false)
            } else if let Some(declaration) = line.strip_prefix("// uzu-array: ") {
                (declaration, true)
            } else {
                continue;
            };
            let (kind, name) = declaration.split_once(' ').expect("compiled Slang ABI declaration");
            let (ty, zero) = match kind {
                "float" => ("f32", "0.0"),
                "bool" => ("bool", "false"),
                "int" => ("i32", "0i"),
                _ => ("u32", "0u"),
            };
            if array {
                let index = array_offsets.get(name).copied().unwrap_or(0);
                let expression = if self.params.get(name).is_some_and(|p| !p.words.is_empty()) {
                    format!("p.data[(i + {index}u) / 4u][(i + {index}u) % 4u]")
                } else {
                    zero.to_owned()
                };
                source.push_str(&format!("fn uzu_array_{name}(i:u32)->{ty}{{return {expression};}}\n"));
            } else {
                let expression = substitutions.get(name).map_or(zero, String::as_str);
                source.push_str(&format!("fn uzu_param_{name}()->{ty}{{return {expression};}}\n"));
            }
        }
        let mut known = BTreeSet::new();
        for (name, binding, dtype, offset) in logical {
            known.insert(name.to_owned());
            source.push_str(&accessors(name, binding, dtype, &offset, physical[binding].is_writable()));
        }
        // Optional buffers have dead-code helpers, allowing the same WGSL body
        // to be specialized without creating dummy GPU resources.
        for token in body.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
            for prefix in ["load_", "store_", "word_", "put_", "round_"] {
                if let Some(name) = token.strip_prefix(prefix)
                    && !known.contains(name)
                {
                    known.insert(name.to_owned());
                    source.push_str(&format!("fn load_{name}(i:u32)->f32{{return 0.0;}}\nfn word_{name}(i:u32)->u32{{return 0u;}}\nfn store_{name}(i:u32,v:f32){{}}\nfn put_{name}(i:u32,v:u32){{}}\nfn round_{name}(v:f32)->f32{{return v;}}\n"));
                }
            }
        }
        source.push_str(&body);
        (source, physical, uniforms, self.groups)
    }
}
fn uniform(index: usize) -> String {
    format!("p.data[{}].{}", index / 4, ["x", "y", "z", "w"][index % 4])
}
fn accessors(
    name: &str,
    b: usize,
    dtype: DataType,
    offset: &str,
    writable: bool,
) -> String {
    let raw = if writable {
        format!("atomicLoad(&b{b}[byte / 4u])")
    } else {
        format!("b{b}[byte / 4u]")
    };
    let bytes = dtype.size_in_bytes();
    let extract = match bytes {
        1 => "(raw >> ((byte % 4u)*8u)) & 255u",
        2 => "(raw >> ((byte % 4u)*8u)) & 65535u",
        _ => "raw",
    };
    let convert = match dtype {
        DataType::BF16 => "bitcast<f32>(v << 16u)",
        DataType::F16 => "unpack2x16float(v).x",
        DataType::F32 => "bitcast<f32>(v)",
        DataType::I8 => "f32(bitcast<i32>(v << 24u) >> 24)",
        DataType::I32 => "f32(bitcast<i32>(v))",
        _ => "f32(v)",
    };
    let bits = match dtype {
        DataType::BF16 => "bf16_bits(v)",
        DataType::F16 => "pack2x16float(vec2<f32>(v,0.0)) & 65535u",
        DataType::F32 => "bitcast<u32>(v)",
        DataType::I8 | DataType::I32 => "bitcast<u32>(i32(v))",
        _ => "u32(v)",
    };
    let round = match dtype {
        DataType::BF16 => "bitcast<f32>(bf16_bits(v)<<16u)",
        DataType::F16 => "unpack2x16float(pack2x16float(vec2<f32>(v,0.0))).x",
        _ => "v",
    };
    let mut s = format!(
        "fn load_{name}(i:u32)->f32{{let byte={offset}+i*{bytes}u;let raw={raw};let v={extract};return {convert};}}\nfn word_{name}(i:u32)->u32{{let byte={offset}+i*4u;return {raw};}}\nfn round_{name}(v:f32)->f32{{return {round};}}\n"
    );
    if writable {
        let store = if bytes < 4 {
            let mask = if bytes == 2 {
                65535
            } else {
                255
            };
            format!(
                "let shift=(byte%4u)*8u; atomicAnd(&b{b}[byte/4u],~({mask}u<<shift)); atomicOr(&b{b}[byte/4u],(bits&{mask}u)<<shift);"
            )
        } else {
            format!("atomicStore(&b{b}[byte/4u],bits);")
        };
        s.push_str(&format!("fn store_{name}(i:u32,v:f32){{let byte={offset}+i*{bytes}u;let bits={bits};{store}}}\nfn put_{name}(i:u32,v:u32){{atomicStore(&b{b}[({offset}/4u)+i],v);}}\n"));
    }
    s
}
const STORAGE_FORMATS: &str = r#"
fn bf16_bits(v:f32)->u32 { let bits=bitcast<u32>(v); if ((bits & 0x7fffffffu)>0x7f800000u) {return (bits>>16u)|64u;} return (bits+0x7fffu+((bits>>16u)&1u))>>16u; }
"#;
