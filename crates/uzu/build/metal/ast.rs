use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

use crate::common::kernel::{Kernel, KernelArgument, KernelArgumentType, KernelParameter, KernelParameterType};

pub type MetalAstNode = clang_ast::Node<MetalAstKind>;

#[derive(Debug, Clone, Deserialize)]
pub struct MetalAstType {
    #[serde(rename = "qualType")]
    qual_type: Box<str>,
    #[serde(rename = "desugaredQualType")]
    desugared_qual_type: Option<Box<str>>,
}

#[derive(Debug, Clone, Deserialize)]
pub enum MetalAstKind {
    TranslationUnitDecl,
    FunctionTemplateDecl,
    TemplateTypeParmDecl {
        name: Option<Box<str>>,
    },
    NonTypeTemplateParmDecl {
        name: Option<Box<str>>,
        #[serde(rename = "type")]
        ty: MetalAstType,
    },
    FunctionDecl {
        name: Box<str>,
    },
    ParmVarDecl {
        name: Option<Box<str>>,
        range: clang_ast::SourceRange,
        #[serde(rename = "type")]
        ty: MetalAstType,
    },
    AnnotateAttr,
    ConstantExpr,
    ImplicitCastExpr,
    StringLiteral {
        value: Box<str>,
    },
    MetalKernelAttr,
    MetalMaxTotalThreadsPerThreadGroupAttr,
    Other,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MetalAddressSpace {
    Device,
    Constant,
    Threadgroup,
    Thread,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MetalDeclarator {
    Value,
    Pointer,
    Reference,
    Array(Option<Box<str>>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MetalBaseType {
    Bool,
    Int,
    UInt,
    Float,
    Simd,
    Named(Box<str>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MetalTypeFacts {
    pub spelling_type: Box<str>,
    pub canonical_type: Box<str>,
    pub base: MetalBaseType,
    pub namespace: Option<Box<[Box<str>]>>,
    pub is_const: bool,
    pub is_volatile: bool,
    pub address_space: Option<MetalAddressSpace>,
    pub declarator: MetalDeclarator,
}

impl MetalTypeFacts {
    pub fn from_clang_types(
        spelling_type: &str,
        desugared_type: Option<&str>,
    ) -> Self {
        let canonical_type = desugared_type.unwrap_or(spelling_type);
        let declarator = Self::parse_declarator(canonical_type);
        let (base, namespace) = Self::parse_base_and_namespace(canonical_type);
        let words = Self::split_words(canonical_type);

        let is_const = words.iter().any(|word| *word == "const");
        let is_volatile = words.iter().any(|word| *word == "volatile");
        let address_space = if words.iter().any(|word| *word == "threadgroup") {
            Some(MetalAddressSpace::Threadgroup)
        } else if words.iter().any(|word| *word == "constant") {
            Some(MetalAddressSpace::Constant)
        } else if words.iter().any(|word| *word == "device") {
            Some(MetalAddressSpace::Device)
        } else if words.iter().any(|word| *word == "thread") {
            Some(MetalAddressSpace::Thread)
        } else {
            None
        };

        Self {
            spelling_type: spelling_type.into(),
            canonical_type: canonical_type.into(),
            base,
            namespace,
            is_const,
            is_volatile,
            address_space,
            declarator,
        }
    }

    fn split_words(c_type: &str) -> Vec<&str> {
        c_type
            .split(|character: char| {
                character.is_whitespace() || matches!(character, '*' | '&' | '[' | ']' | '(' | ')' | ',')
            })
            .filter(|word| !word.is_empty())
            .collect()
    }

    fn parse_declarator(c_type: &str) -> MetalDeclarator {
        if let Some(array_size) = Self::parse_array_size(c_type) {
            return MetalDeclarator::Array(array_size);
        }

        if c_type.contains('&') {
            return MetalDeclarator::Reference;
        }

        if c_type.contains('*') {
            return MetalDeclarator::Pointer;
        }

        MetalDeclarator::Value
    }

    fn parse_array_size(c_type: &str) -> Option<Option<Box<str>>> {
        let left_bracket_index = c_type.rfind('[')?;
        let right_bracket_index = c_type[left_bracket_index..].find(']')? + left_bracket_index;
        if right_bracket_index <= left_bracket_index {
            return Some(None);
        }

        let size_expression = c_type[left_bracket_index + 1..right_bracket_index].trim();
        if size_expression.is_empty() {
            Some(None)
        } else {
            Some(Some(size_expression.into()))
        }
    }

    fn parse_base_and_namespace(c_type: &str) -> (MetalBaseType, Option<Box<[Box<str>]>>) {
        let type_without_array = if let Some(left_bracket_index) = c_type.rfind('[') {
            &c_type[..left_bracket_index]
        } else {
            c_type
        };

        let type_without_declarator = type_without_array
            .chars()
            .map(|character| if matches!(character, '*' | '&') { ' ' } else { character })
            .collect::<String>();

        let filtered_words = type_without_declarator
            .split_whitespace()
            .filter(|word| {
                !matches!(
                    *word,
                    "const" | "volatile" | "device" | "threadgroup" | "constant" | "thread" | "static"
                )
            })
            .collect::<Vec<_>>();

        let base_candidate = filtered_words.join(" ");

        match base_candidate.as_str() {
            "bool" => (MetalBaseType::Bool, None),
            "int" | "int32_t" => (MetalBaseType::Int, None),
            "uint" | "uint32_t" | "unsigned" | "unsigned int" => (MetalBaseType::UInt, None),
            "float" => (MetalBaseType::Float, None),
            "Simd" => (MetalBaseType::Simd, None),
            _ => {
                let segments = base_candidate
                    .split("::")
                    .map(str::trim)
                    .filter(|segment| !segment.is_empty())
                    .collect::<Vec<_>>();

                if segments.is_empty() {
                    return (MetalBaseType::Named(base_candidate.into()), None);
                }

                let namespace = segments
                    .iter()
                    .take(segments.len().saturating_sub(1))
                    .map(|segment| (*segment).to_string().into_boxed_str())
                    .collect::<Vec<_>>();
                let type_name = segments.last().unwrap().to_string().into_boxed_str();
                if namespace.is_empty() {
                    (MetalBaseType::Named(type_name), None)
                } else {
                    (MetalBaseType::Named(type_name), Some(namespace.into_boxed_slice()))
                }
            },
        }
    }

    pub fn to_rust_scalar_type(&self) -> anyhow::Result<Box<str>> {
        let rust_type = match &self.base {
            MetalBaseType::Bool => "bool".into(),
            MetalBaseType::Int => "i32".into(),
            MetalBaseType::UInt => "u32".into(),
            MetalBaseType::Float => "f32".into(),
            MetalBaseType::Simd => bail!("Simd type cannot be converted to Rust scalar type"),
            MetalBaseType::Named(name) => {
                if let Some(namespace_segments) = &self.namespace
                    && namespace_segments.len() == 1
                    && namespace_segments[0].as_ref() == "uzu"
                {
                    return Ok(format!("crate::backends::common::gpu_types::{}", name).into());
                }
                name.clone()
            },
        };

        Ok(rust_type)
    }

    pub fn to_specialization_type_name(&self) -> Box<str> {
        match &self.base {
            MetalBaseType::Bool => "bool".into(),
            MetalBaseType::Int => "int32_t".into(),
            MetalBaseType::UInt => "uint32_t".into(),
            MetalBaseType::Float => "float".into(),
            MetalBaseType::Simd => "Simd".into(),
            MetalBaseType::Named(name) => {
                if let Some(namespace_segments) = &self.namespace {
                    let namespace = namespace_segments.iter().map(|segment| segment.as_ref()).collect::<Vec<_>>();
                    format!("{}::{}", namespace.join("::"), name).into()
                } else {
                    name.clone()
                }
            },
        }
    }

    pub fn is_simd(&self) -> bool {
        matches!(self.base, MetalBaseType::Simd)
    }

    pub fn is_buffer(&self) -> bool {
        matches!(self.address_space, Some(MetalAddressSpace::Device))
            && matches!(self.declarator, MetalDeclarator::Pointer)
    }

    pub fn is_constant_scalar(&self) -> bool {
        self.is_const
            && matches!(self.address_space, Some(MetalAddressSpace::Constant))
            && matches!(self.declarator, MetalDeclarator::Reference)
    }

    pub fn is_constant_array(&self) -> bool {
        self.is_const
            && matches!(self.address_space, Some(MetalAddressSpace::Constant))
            && matches!(self.declarator, MetalDeclarator::Pointer)
    }

    pub fn is_threadgroup(&self) -> bool {
        matches!(self.address_space, Some(MetalAddressSpace::Threadgroup))
            && matches!(
                self.declarator,
                MetalDeclarator::Pointer | MetalDeclarator::Reference | MetalDeclarator::Array(_)
            )
    }
}

fn annotation_from_ast_node(annotation_node: MetalAstNode) -> anyhow::Result<Box<[Box<str>]>> {
    if !matches!(annotation_node.kind, MetalAstKind::AnnotateAttr) {
        bail!(
            "unexpected kind of root node: MetalAstKind::AnnotateAttr expected, but {:?} found",
            annotation_node.kind
        );
    }

    annotation_node
        .inner
        .into_iter()
        .map(|mut constant_expr| {
            let MetalAstKind::ConstantExpr = constant_expr.kind else {
                bail!("expected ConstantExpr, found {:?}", constant_expr.kind);
            };

            if constant_expr.inner.len() != 1 {
                bail!("ConstantExpr must have exactly one child, found {}", constant_expr.inner.len());
            }

            let mut implicit_cast_expr = constant_expr.inner.pop().unwrap();

            let MetalAstKind::ImplicitCastExpr = implicit_cast_expr.kind else {
                bail!("expected ImplicitCastExpr, found {:?}", implicit_cast_expr.kind);
            };

            if implicit_cast_expr.inner.len() != 1 {
                bail!("ImplicitCastExpr must have exactly one child, found {}", implicit_cast_expr.inner.len());
            }

            let string_literal = implicit_cast_expr.inner.pop().unwrap();

            let MetalAstKind::StringLiteral {
                value,
            } = string_literal.kind
            else {
                bail!("expected StringLiteral, found {:?}", string_literal.kind);
            };

            // NOTE: string literal includes "" (and is probably not parsed?), using json parse here for now
            Ok(serde_json::from_str(&value).context("failed to parse string literal")?)
        })
        .collect()
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MetalConstantType {
    Scalar,
    Array,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MetalGroupsType {
    Direct(Box<str>),
    Indirect,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MetalArgumentType {
    Buffer,
    Constant((Box<str>, MetalConstantType)),
    Shared(Option<Box<str>>),
    Specialize(Box<str>),
    Axis(Box<str>, Box<str>),
    Groups(MetalGroupsType),
    Threads(Box<str>),
    Simd,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetalArgument {
    pub name: Box<str>,
    pub c_type: Box<str>,
    pub type_facts: MetalTypeFacts,
    pub annotation: Option<Box<[Box<str>]>>,
    pub source: Box<str>,
}

impl MetalArgument {
    fn scalar_type_to_rust(c_type: &str) -> anyhow::Result<Box<str>> {
        MetalTypeFacts::from_clang_types(c_type, None)
            .to_rust_scalar_type()
            .with_context(|| format!("failed to parse scalar type: {}", c_type))
    }

    fn from_ast_node_and_source(
        argument_node: MetalAstNode,
        source: &str,
    ) -> anyhow::Result<Self> {
        let MetalAstKind::ParmVarDecl {
            name,
            range,
            ty: MetalAstType {
                qual_type,
                desugared_qual_type,
            },
        } = argument_node.kind
        else {
            bail!("argument isn't ParmVarDecl: {:?}", argument_node.kind);
        };

        let name = name.context("ParmVarDecl has no name")?;

        let type_facts = MetalTypeFacts::from_clang_types(&qual_type, desugared_qual_type.as_deref());
        let c_type = desugared_qual_type.unwrap_or(qual_type);

        if argument_node.inner.len() > 1 {
            bail!("more than one annotation on argument ast node");
        }

        let annotation = if let Some(annotation_node) = argument_node.inner.first() {
            Some(annotation_from_ast_node(annotation_node.clone())?)
        } else {
            None
        };

        let start_offset = range.begin.spelling_loc.context("no start location in source range")?.offset;
        let end_offset = range.end.spelling_loc.context("no end location in source range")?.offset;
        let source =
            str::from_utf8(&source.as_bytes()[start_offset..=end_offset]).context("source range is not utf-8")?.into();

        Ok(Self {
            name,
            c_type,
            type_facts,
            annotation,
            source,
        })
    }

    pub fn argument_condition(&self) -> anyhow::Result<Option<&str>> {
        if let Some(annotation) = self.annotation.as_ref()
            && annotation.first().map(|annotation_item| annotation_item.as_ref()) == Some("dsl.optional")
        {
            assert!(
                matches!(self.argument_type().unwrap(), MetalArgumentType::Buffer | MetalArgumentType::Constant(_)),
                "Only a buffer or a constant can be optional"
            );
            if annotation.len() != 2 {
                bail!("dsl.optional takes 1 argument, found {}", annotation.len() - 1);
            }
            Ok(Some(annotation[1].as_ref()))
        } else {
            Ok(None)
        }
    }

    pub fn argument_type(&self) -> anyhow::Result<MetalArgumentType> {
        const USE_TYPE_STRING_FALLBACK: bool = true;

        if let Some(annotation) = self.annotation.as_ref()
            && annotation.first().map(|annotation_item| annotation_item.as_ref()) != Some("dsl.optional")
        {
            let mut annotation = annotation.to_vec();

            if annotation.is_empty() {
                bail!("empty annotation");
            }
            let annotation_key = annotation.remove(0);

            match &*annotation_key {
                "dsl.specialize" => {
                    if !annotation.is_empty() {
                        bail!("dsl.specialize takes no arguments, got {}", annotation.len());
                    }
                    let rust_type = Self::scalar_type_to_rust(&self.c_type)?;
                    Ok(MetalArgumentType::Specialize(rust_type.into()))
                },
                "dsl.axis" => {
                    if annotation.len() != 2 {
                        bail!("dsl.axis requires 2 arguments, got {}", annotation.len());
                    }
                    Ok(MetalArgumentType::Axis(annotation.remove(0), annotation.remove(0)))
                },
                "dsl.groups" => {
                    if annotation.len() != 1 {
                        bail!("dsl.groups requires 1 argument, got {}", annotation.len());
                    }
                    let dimension = annotation.remove(0);
                    match dimension.as_ref() {
                        "INDIRECT" => Ok(MetalArgumentType::Groups(MetalGroupsType::Indirect)),
                        _ => Ok(MetalArgumentType::Groups(MetalGroupsType::Direct(dimension))),
                    }
                },
                "dsl.threads" => {
                    if annotation.len() != 1 {
                        bail!("dsl.threads requires 1 argument, got {}", annotation.len());
                    }
                    Ok(MetalArgumentType::Threads(annotation.remove(0)))
                },
                _ => bail!("unknown annotation: {annotation_key}"),
            }
        } else {
            if self.type_facts.is_simd() {
                return Ok(MetalArgumentType::Simd);
            }

            if self.type_facts.is_buffer() {
                return Ok(MetalArgumentType::Buffer);
            }

            if self.type_facts.is_constant_scalar() {
                let rust_type = self.type_facts.to_rust_scalar_type()?;
                return Ok(MetalArgumentType::Constant((rust_type, MetalConstantType::Scalar)));
            }

            if self.type_facts.is_constant_array() {
                let rust_type = self.type_facts.to_rust_scalar_type()?;
                return Ok(MetalArgumentType::Constant((rust_type, MetalConstantType::Array)));
            }

            if self.type_facts.is_threadgroup() {
                let size_expression = match &self.type_facts.declarator {
                    MetalDeclarator::Array(Some(size)) => Some(size.clone()),
                    MetalDeclarator::Pointer => {
                        if self.source.contains('[') {
                            let left_bracket_index =
                                self.source.rfind('[').context("threadgroup missing size bracket")? + 1;
                            let right_bracket_index =
                                self.source.rfind(']').context("threadgroup missing size bracket")?;
                            let size_substring = &self.source[left_bracket_index..right_bracket_index];
                            Some(size_substring.into())
                        } else {
                            None
                        }
                    },
                    MetalDeclarator::Reference => None,
                    _ => None,
                };
                return Ok(MetalArgumentType::Shared(size_expression));
            }

            if USE_TYPE_STRING_FALLBACK {
                return self.argument_type_fallback();
            }

            bail!("cannot classify c type: {} (facts={:?})", self.c_type, self.type_facts);
        }
    }

    fn argument_type_fallback(&self) -> anyhow::Result<MetalArgumentType> {
        if self.c_type.as_ref() == "Simd" || self.c_type.as_ref() == "const Simd" {
            Ok(MetalArgumentType::Simd)
        } else if self.c_type.contains("device") && self.c_type.contains('*') && !self.c_type.contains('&') {
            Ok(MetalArgumentType::Buffer)
        } else if let ["const", "constant", scalar_c_type, "&"] =
            self.c_type.split_whitespace().collect::<Vec<_>>().as_slice()
        {
            Ok(MetalArgumentType::Constant((
                Self::scalar_type_to_rust(scalar_c_type)?.into(),
                MetalConstantType::Scalar,
            )))
        } else if let ["const", "constant", scalar_c_type, "*"] =
            self.c_type.split_whitespace().collect::<Vec<_>>().as_slice()
        {
            Ok(MetalArgumentType::Constant((
                Self::scalar_type_to_rust(scalar_c_type)?.into(),
                MetalConstantType::Array,
            )))
        } else if self.c_type.contains("threadgroup") && self.c_type.contains('*') {
            let left_bracket_index = self.source.rfind('[').context("threadgroup missing size bracket")? + 1;
            let right_bracket_index = self.source.rfind(']').context("threadgroup missing size bracket")?;
            let size_expression = &self.source[left_bracket_index..right_bracket_index];
            Ok(MetalArgumentType::Shared(Some(size_expression.into())))
        } else if self.c_type.contains("threadgroup") && self.c_type.contains('&') {
            Ok(MetalArgumentType::Shared(None))
        } else {
            bail!("cannot parse c type (fallback): {}", self.c_type);
        }
    }

    fn to_parameter(&self) -> Option<KernelParameter> {
        match self.argument_type().unwrap() {
            MetalArgumentType::Specialize(rust_type) => Some(KernelParameter {
                name: self.name.clone(),
                ty: KernelParameterType::Value(rust_type),
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MetalTemplateParameterType {
    Type,
    Value(Box<str>),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetalTemplateParameter {
    pub name: Box<str>,
    pub ty: MetalTemplateParameterType,
    pub variants: Box<[Box<str>]>,
}

impl MetalTemplateParameter {
    fn to_parameter(&self) -> KernelParameter {
        KernelParameter {
            name: self.name.clone(),
            ty: match &self.ty {
                MetalTemplateParameterType::Type => KernelParameterType::Type,
                MetalTemplateParameterType::Value(rust_type) => KernelParameterType::Value(rust_type.clone()),
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetalKernelInfo {
    pub name: Box<str>,
    pub arguments: Box<[MetalArgument]>,
    pub variants: Option<Box<[MetalTemplateParameter]>>,
}

impl MetalKernelInfo {
    pub fn has_axis(&self) -> bool {
        self.arguments.iter().any(|argument| matches!(argument.argument_type(), Ok(MetalArgumentType::Axis(..))))
    }

    pub fn has_groups(&self) -> bool {
        self.arguments.iter().any(|argument| matches!(argument.argument_type(), Ok(MetalArgumentType::Groups(_))))
    }

    pub fn has_groups_direct(&self) -> bool {
        self.arguments.iter().any(|argument| {
            matches!(argument.argument_type(), Ok(MetalArgumentType::Groups(MetalGroupsType::Direct(_))))
        })
    }

    pub fn has_groups_indirect(&self) -> bool {
        self.arguments.iter().any(|argument| {
            matches!(argument.argument_type(), Ok(MetalArgumentType::Groups(MetalGroupsType::Indirect)))
        })
    }

    pub fn has_threads(&self) -> bool {
        self.arguments.iter().any(|argument| matches!(argument.argument_type(), Ok(MetalArgumentType::Threads(_))))
    }

    pub fn has_simd(&self) -> bool {
        self.arguments.iter().any(|argument| matches!(argument.argument_type(), Ok(MetalArgumentType::Simd)))
    }

    pub fn to_kernel(&self) -> Kernel {
        let mut indirect_flag = false;

        Kernel {
            name: self.name.clone(),
            parameters: self
                .variants
                .as_ref()
                .map(|template_parameters| {
                    template_parameters
                        .iter()
                        .map(|template_parameter| template_parameter.to_parameter())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
                .into_iter()
                .chain(self.arguments.iter().filter_map(|argument| argument.to_parameter()))
                .collect(),
            arguments: self
                .arguments
                .iter()
                .filter_map(|argument| match argument.argument_type() {
                    Ok(MetalArgumentType::Buffer) => Some(KernelArgument {
                        name: argument.name.clone(),
                        conditional: argument.argument_condition().unwrap().is_some(),
                        ty: KernelArgumentType::Buffer,
                    }),
                    Ok(MetalArgumentType::Groups(MetalGroupsType::Indirect)) if !indirect_flag => {
                        indirect_flag = true;
                        Some(KernelArgument {
                            name: "__dsl_indirect_dispatch_buffer".into(),
                            conditional: false,
                            ty: KernelArgumentType::Buffer,
                        })
                    },
                    Ok(MetalArgumentType::Constant((ty, MetalConstantType::Scalar))) => Some(KernelArgument {
                        name: argument.name.clone(),
                        conditional: argument.argument_condition().unwrap().is_some(),
                        ty: KernelArgumentType::Scalar(ty),
                    }),
                    Ok(MetalArgumentType::Constant((ty, MetalConstantType::Array))) => Some(KernelArgument {
                        name: argument.name.clone(),
                        conditional: argument.argument_condition().unwrap().is_some(),
                        ty: KernelArgumentType::Constant(ty),
                    }),
                    _ => None,
                })
                .collect(),
        }
    }
}

impl MetalKernelInfo {
    pub fn from_ast_node_and_source(
        node: MetalAstNode,
        source: &str,
    ) -> anyhow::Result<Option<Self>> {
        let (is_template, template_parameters, node) = if matches!(node.kind, MetalAstKind::FunctionTemplateDecl) {
            let mut template_parameters = Vec::new();
            let mut function_node = None;

            for child in node.inner {
                match child.kind {
                    MetalAstKind::TemplateTypeParmDecl {
                        name,
                    } => {
                        let name = name.context("template parameter missing name")?;
                        template_parameters.push((name, None));
                    },
                    MetalAstKind::NonTypeTemplateParmDecl {
                        name,
                        ty,
                    } => {
                        let name = name.context("template parameter missing name")?;
                        template_parameters.push((name, Some(ty)));
                    },
                    MetalAstKind::FunctionDecl {
                        name: _,
                    } => {
                        function_node = Some(child);
                    },
                    _ => (),
                }
            }

            let node = function_node.context("unexpected kind of root node: template without function")?;

            (true, template_parameters, node)
        } else if matches!(node.kind, MetalAstKind::FunctionDecl { .. }) {
            (false, Vec::new(), node)
        } else {
            return Ok(None);
        };

        let MetalAstKind::FunctionDecl {
            name,
        } = node.kind
        else {
            bail!("unexpected kind of root node: function expected, but {:?} found", node.kind);
        };

        let mut argument_nodes = Vec::new();
        let mut annotations = Vec::new();

        for node in node.inner {
            match node.kind {
                MetalAstKind::ParmVarDecl {
                    name: _,
                    range: _,
                    ty: _,
                } => argument_nodes.push(node),
                MetalAstKind::AnnotateAttr => annotations.push(annotation_from_ast_node(node)?),
                _ => (),
            }
        }

        let annotations = annotations
            .into_iter()
            .map(|annotation_values| {
                if !annotation_values.is_empty() {
                    let mut annotation_values = annotation_values.into_vec();
                    Ok((annotation_values.remove(0), annotation_values.into_boxed_slice()))
                } else {
                    bail!("zero length annotation");
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        if !annotations.iter().any(|(annotation_key, _)| annotation_key.as_ref() == "dsl.kernel") {
            return Ok(None);
        }

        let variants: Box<[_]> = annotations
            .iter()
            .filter(|(annotation_key, _)| annotation_key.as_ref() == "dsl.variants")
            .map(|(_, annotation_values)| {
                let [variant_name, variant_values] = annotation_values.as_ref() else {
                    bail!("malformed dsl.variants annotation");
                };

                let variant_values = variant_values
                    .split(',')
                    .map(|variant_value| variant_value.trim().into())
                    .collect::<Box<[Box<str>]>>();

                Ok((variant_name.clone(), variant_values))
            })
            .collect::<anyhow::Result<_>>()?;

        if variants.is_empty() != !is_template {
            bail!("mismatch between AST nodes and variants annotation");
        }

        let variants = if is_template {
            let template_names = template_parameters.iter().map(|(name, _)| name.as_ref()).collect::<Vec<_>>();
            let variant_names = variants.iter().map(|(name, _)| name.as_ref()).collect::<Vec<_>>();
            if template_names != variant_names {
                bail!("template parameters {:?} do not match dsl.variants order {:?}", template_names, variant_names);
            }

            Some(
                template_parameters
                    .into_iter()
                    .zip(variants.into_iter())
                    .map(|((name, ty), (variant_name, variants))| {
                        assert_eq!(name, variant_name);

                        Ok(MetalTemplateParameter {
                            name,
                            ty: match ty {
                                None => MetalTemplateParameterType::Type,
                                Some(non_type_template_type) => {
                                    MetalTemplateParameterType::Value(MetalArgument::scalar_type_to_rust(
                                        non_type_template_type
                                            .desugared_qual_type
                                            .unwrap_or(non_type_template_type.qual_type)
                                            .as_ref(),
                                    )?)
                                },
                            },
                            variants,
                        })
                    })
                    .collect::<anyhow::Result<_>>()?,
            )
        } else {
            None
        };

        let arguments = argument_nodes
            .into_iter()
            .map(|argument_node| MetalArgument::from_ast_node_and_source(argument_node, source))
            .collect::<anyhow::Result<Box<[_]>>>()?;

        Ok(Some(MetalKernelInfo {
            name,
            arguments,
            variants,
        }))
    }
}

pub fn validate_raw_kernel(node: &MetalAstNode) -> anyhow::Result<()> {
    let (node, is_template) = if matches!(node.kind, MetalAstKind::FunctionTemplateDecl) {
        let inner_function = node.inner.iter().find(|child| {
            matches!(
                child.kind,
                MetalAstKind::FunctionDecl {
                    name: _
                }
            )
        });
        match inner_function {
            Some(function_node) => (function_node, true),
            None => return Ok(()),
        }
    } else if matches!(node.kind, MetalAstKind::FunctionDecl { .. }) {
        (node, false)
    } else {
        return Ok(());
    };

    let MetalAstKind::FunctionDecl {
        name,
    } = &node.kind
    else {
        return Ok(());
    };

    let mut has_metal_kernel_attr = false;
    let mut has_max_threads_attr = false;
    let mut has_dsl_kernel = false;

    for child in &node.inner {
        match &child.kind {
            MetalAstKind::MetalKernelAttr => has_metal_kernel_attr = true,
            MetalAstKind::MetalMaxTotalThreadsPerThreadGroupAttr => has_max_threads_attr = true,
            MetalAstKind::AnnotateAttr => {
                if let Ok(annotation) = annotation_from_ast_node(child.clone()) {
                    if annotation.first().map(|annotation_item| annotation_item.as_ref()) == Some("dsl.kernel") {
                        has_dsl_kernel = true;
                    }
                }
            },
            _ => (),
        }
    }

    if has_metal_kernel_attr && !has_dsl_kernel && !has_max_threads_attr {
        let kind = if is_template {
            "template"
        } else {
            "function"
        };
        bail!(
            "kernel {kind} '{name}' is missing `[[max_total_threads_per_threadgroup(N)]]` attribute! it **MUST** be present and correct!"
        );
    }

    Ok(())
}
