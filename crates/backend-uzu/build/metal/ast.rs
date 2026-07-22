use std::collections::HashMap;

use anyhow::{Context, bail};
use quote::quote;
use serde::{Deserialize, Serialize};

use crate::common::{
    enum_paths::EnumPaths,
    expr_rewrite::rewrite_paths_with,
    identifiers::{ArgumentName, KernelName},
    kernel::{Kernel, KernelArgument, KernelArgumentType, KernelBufferAccess, KernelParameter, KernelParameterType},
};

pub type MetalAstNode = clang_ast::Node<MetalAstKind>;
type IntegerObjectDefines = Box<[(Box<str>, u64)]>;

fn integer_object_defines_from_source(source: &str) -> IntegerObjectDefines {
    source
        .lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix("#define")?.trim_start();
            let (name, value) = rest.split_once(char::is_whitespace)?;
            let value = value.split_whitespace().next()?.trim_end_matches(['u', 'U']).parse::<u64>().ok()?;
            Some((name.into(), value))
        })
        .collect()
}

fn expand_integer_defines_in_threadgroup_dimension(
    dimension_expression: &str,
    defines: &IntegerObjectDefines,
) -> Box<str> {
    let Ok(mut expr) = syn::parse_str::<syn::Expr>(dimension_expression) else {
        return dimension_expression.into();
    };
    rewrite_paths_with(&mut expr, |path| {
        let ident = path.get_ident()?;
        let (_, value) = defines.iter().find(|(name, _)| ident == name.as_ref())?;
        syn::parse_str::<syn::Expr>(&value.to_string()).ok()
    });
    quote!(#expr).to_string().into_boxed_str()
}

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
    Other,
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
            serde_json::from_str(&value).context("failed to parse string literal")
        })
        .collect()
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MetalBufferAccess {
    Read,
    ReadWrite,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetalConstantType {
    Scalar,
    Array(Option<Box<str>>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetalGroupsType {
    Direct(Box<str>),
    Indirect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetalArgumentType {
    Buffer(MetalBufferAccess),
    Constant((Box<str>, MetalConstantType)),
    Shared(Option<Box<str>>),
    Specialize(Box<str>),
    Axis(Box<str>, Box<str>),
    Groups(MetalGroupsType),
    Threads(Box<str>),
    ThreadContext,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetalArgument {
    pub name: ArgumentName,
    pub c_type: Box<str>,
    pub argument_type: MetalArgumentType,
    pub condition: Option<Box<str>>,
    pub source: Box<str>,
}

pub fn shared_element_type(c_type: &str) -> &str {
    c_type.split(['*', '&', '(']).next().unwrap_or_default().trim_end()
}

pub fn shared_element_byte_size(c_type: &str) -> anyhow::Result<usize> {
    let element_type = shared_element_type(c_type).rsplit(' ').next().unwrap_or_default();

    // Split the lane digit off vectors: "float4" -> "float", 4.
    let (scalar_name, lanes) = match element_type.chars().last() {
        Some(lane_digit @ '2'..='4') => {
            (element_type.strip_suffix(lane_digit).unwrap(), lane_digit.to_digit(10).unwrap() as usize)
        },
        _ => (element_type, 1),
    };

    let scalar_size = match scalar_name {
        "bool" | "char" | "uchar" => 1,
        "short" | "ushort" | "half" | "bfloat" => 2,
        "int" | "uint" | "float" => 4,
        "long" | "ulong" => 8,
        other => bail!("unsupported OPTIONAL threadgroup element type `{other}`"),
    };

    // MSL pads 3-component vectors to 4 lanes, so `float3` is 16 bytes, not 12.
    let padded_lanes = if lanes == 3 {
        4
    } else {
        lanes
    };
    Ok(scalar_size * padded_lanes)
}

impl MetalArgument {
    fn scalar_type_to_rust(c_type: &str) -> anyhow::Result<Box<str>> {
        let mut tokens: Vec<_> = c_type.split_whitespace().collect();
        if tokens.first() == Some(&"const") {
            tokens.remove(0);
        }
        match tokens.as_slice() {
            ["bool"] => Ok("bool".into()),
            ["uint"] | ["uint32_t"] | ["unsigned", "int"] => Ok("u32".into()),
            ["int"] | ["int32_t"] => Ok("i32".into()),
            ["float"] => Ok("f32".into()),
            [vpath] if vpath.starts_with("uzu::") => {
                Ok(vpath.replacen("uzu::", "crate::backends::common::gpu_types::", 1).into())
            },
            _ => bail!("unknown scalar type: {c_type}"),
        }
    }

    fn from_ast_node_and_source(
        argument_node: MetalAstNode,
        source: &str,
        integer_defines: &IntegerObjectDefines,
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
        let source: Box<str> =
            str::from_utf8(&source.as_bytes()[start_offset..=end_offset]).context("source range is not utf-8")?.into();

        let (argument_type, condition) =
            parse_argument_annotation(&c_type, &source, annotation.as_deref(), integer_defines)?;

        Ok(Self {
            name: ArgumentName::from(name),
            c_type,
            argument_type,
            condition,
            source,
        })
    }

    fn to_parameter(&self) -> Option<KernelParameter> {
        match &self.argument_type {
            MetalArgumentType::Specialize(ty) => Some(KernelParameter {
                name: Box::from(&*self.name),
                ty: KernelParameterType::Value(ty.clone()),
            }),
            _ => None,
        }
    }

    pub fn is_optional_shared(&self) -> bool {
        matches!(self.argument_type, MetalArgumentType::Shared(_)) && self.condition.is_some()
    }
}

fn parse_argument_annotation(
    c_type: &str,
    source: &str,
    annotation: Option<&[Box<str>]>,
    integer_defines: &IntegerObjectDefines,
) -> anyhow::Result<(MetalArgumentType, Option<Box<str>>)> {
    if let Some(annotation) = annotation
        && annotation.first().map(|s| s.as_ref()) == Some("dsl.optional")
    {
        if annotation.len() != 2 {
            bail!("dsl.optional takes 1 argument, found {}", annotation.len() - 1);
        }
        let argument_type = parse_argument_type(c_type, source, None, integer_defines)?;
        if !matches!(
            argument_type,
            MetalArgumentType::Buffer(_) | MetalArgumentType::Constant(_) | MetalArgumentType::Shared(_)
        ) {
            bail!("Only a buffer, a constant or a threadgroup argument can be optional");
        }
        return Ok((argument_type, Some(annotation[1].clone())));
    }

    let argument_type = parse_argument_type(c_type, source, annotation, integer_defines)?;
    Ok((argument_type, None))
}

fn parse_argument_type(
    c_type: &str,
    source: &str,
    annotation: Option<&[Box<str>]>,
    integer_defines: &IntegerObjectDefines,
) -> anyhow::Result<MetalArgumentType> {
    if let Some(annotation) = annotation {
        let mut annotation = annotation.to_vec();
        if annotation.is_empty() {
            bail!("empty annotation");
        }
        let annotation_key = annotation.remove(0);

        return match &*annotation_key {
            "dsl.specialize" => {
                if !annotation.is_empty() {
                    bail!("dsl.specialize takes no arguments, got {}", annotation.len());
                }
                let rust_type = MetalArgument::scalar_type_to_rust(c_type)?;
                Ok(MetalArgumentType::Specialize(rust_type))
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
                let dim = annotation.remove(0);
                match dim.as_ref() {
                    "INDIRECT" => Ok(MetalArgumentType::Groups(MetalGroupsType::Indirect)),
                    _ => Ok(MetalArgumentType::Groups(MetalGroupsType::Direct(dim))),
                }
            },
            "dsl.threads" => {
                if annotation.len() != 1 {
                    bail!("dsl.threads requires 1 argument, got {}", annotation.len());
                }
                Ok(MetalArgumentType::Threads(annotation.remove(0)))
            },
            _ => bail!("unknown annotation: {annotation_key}"),
        };
    }

    if c_type == "ThreadContext" || c_type == "const ThreadContext" {
        return Ok(MetalArgumentType::ThreadContext);
    }

    if c_type.contains("device") && c_type.contains('*') && !c_type.contains('&') {
        return Ok(MetalArgumentType::Buffer(if c_type.contains("const") {
            MetalBufferAccess::Read
        } else {
            MetalBufferAccess::ReadWrite
        }));
    }

    if let ["const", "constant", c_type_scalar, "&"] = c_type.split_whitespace().collect::<Vec<_>>().as_slice() {
        return Ok(MetalArgumentType::Constant((
            MetalArgument::scalar_type_to_rust(c_type_scalar)?,
            MetalConstantType::Scalar,
        )));
    }

    if let ["const", "constant", c_type_scalar, "*"] = c_type.split_whitespace().collect::<Vec<_>>().as_slice() {
        let size = if source.contains('[') && source.contains(']') {
            let lbracket = source.rfind('[').context("sized constant missing size bracket")? + 1;
            let rbracket = source.rfind(']').context("sized constant missing size bracket")?;
            Some(source[lbracket..rbracket].into())
        } else {
            None
        };

        return Ok(MetalArgumentType::Constant((
            MetalArgument::scalar_type_to_rust(c_type_scalar)?,
            MetalConstantType::Array(size),
        )));
    }

    if c_type.contains("threadgroup") && c_type.contains('*') {
        let lbracket = source.find('[').context("threadgroup missing size bracket")? + 1;
        let rbracket = source.rfind(']').context("threadgroup missing size bracket")?;
        return Ok(MetalArgumentType::Shared(Some(expand_integer_defines_in_threadgroup_dimension(
            &source[lbracket..rbracket],
            integer_defines,
        ))));
    }

    if c_type.contains("threadgroup") && c_type.contains('&') {
        return Ok(MetalArgumentType::Shared(None));
    }

    bail!("cannot parse c type: {}", c_type);
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

/// The value set for an axis whose `VARIANTS` list the shader left out: a bool ranges
/// over both values, an enum over every member of its Rust definition. Restating either
/// in the shader only creates a second place for them to be wrong.
///
/// Type and numeric parameters have no derivable set and must be declared.
fn default_variants(
    name: &str,
    ty: &MetalTemplateParameterType,
    enum_paths: &EnumPaths,
) -> anyhow::Result<Box<[Box<str>]>> {
    let MetalTemplateParameterType::Value(rust_type) = ty else {
        bail!("template parameter `{name}` is a type, so it needs an explicit VARIANTS list");
    };

    if rust_type.as_ref() == "bool" {
        return Ok(Box::new(["false".into(), "true".into()]));
    }

    let short_name = rust_type.rsplit("::").next().unwrap_or(rust_type);
    match enum_paths.variants_for(short_name) {
        Some(variants) if !variants.is_empty() => {
            Ok(variants.iter().map(|(variant, _)| format!("{short_name}::{variant}").into()).collect())
        },
        _ => bail!("template parameter `{name}` has no derivable value set, so it needs an explicit VARIANTS list"),
    }
}

impl MetalTemplateParameter {
    fn to_parameter(&self) -> KernelParameter {
        KernelParameter {
            name: self.name.clone(),
            ty: match &self.ty {
                MetalTemplateParameterType::Type => KernelParameterType::Type,
                MetalTemplateParameterType::Value(ty) => KernelParameterType::Value(ty.clone()),
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetalKernelInfo {
    pub public: bool,
    pub name: KernelName,
    pub arguments: Box<[MetalArgument]>,
    pub variants: Option<Box<[MetalTemplateParameter]>>,
    pub constraints: Box<[Box<str>]>,
}

impl MetalKernelInfo {
    pub fn has_axis(&self) -> bool {
        self.arguments.iter().any(|a| matches!(&a.argument_type, MetalArgumentType::Axis(..)))
    }

    pub fn has_groups(&self) -> bool {
        self.arguments.iter().any(|a| matches!(&a.argument_type, MetalArgumentType::Groups(_)))
    }

    pub fn has_groups_direct(&self) -> bool {
        self.arguments.iter().any(|a| matches!(&a.argument_type, MetalArgumentType::Groups(MetalGroupsType::Direct(_))))
    }

    pub fn has_groups_indirect(&self) -> bool {
        self.arguments.iter().any(|a| matches!(&a.argument_type, MetalArgumentType::Groups(MetalGroupsType::Indirect)))
    }

    pub fn has_threads(&self) -> bool {
        self.arguments.iter().any(|a| matches!(&a.argument_type, MetalArgumentType::Threads(_)))
    }

    pub fn has_thread_context(&self) -> bool {
        self.arguments.iter().any(|a| matches!(&a.argument_type, MetalArgumentType::ThreadContext))
    }

    pub fn to_kernel(&self) -> Option<Kernel> {
        if !self.public {
            return None;
        }

        let mut indirect_flag = false;

        Some(Kernel {
            name: self.name.clone(),
            parameters: self
                .variants
                .as_ref()
                .map(|v| v.iter().map(|p| p.to_parameter()).collect::<Vec<_>>())
                .unwrap_or_default()
                .into_iter()
                .chain(self.arguments.iter().filter_map(|a| a.to_parameter()))
                .collect(),
            arguments: self
                .arguments
                .iter()
                .filter_map(|a| match &a.argument_type {
                    MetalArgumentType::Buffer(access) => Some(KernelArgument {
                        name: a.name.clone(),
                        conditional: a.condition.is_some(),
                        ty: KernelArgumentType::Buffer(match access {
                            MetalBufferAccess::Read => KernelBufferAccess::Read,
                            MetalBufferAccess::ReadWrite => KernelBufferAccess::ReadWrite,
                        }),
                    }),
                    MetalArgumentType::Groups(MetalGroupsType::Indirect) if !indirect_flag => {
                        indirect_flag = true;
                        Some(KernelArgument {
                            name: "__dsl_indirect_dispatch_buffer".into(),
                            conditional: false,
                            ty: KernelArgumentType::Buffer(KernelBufferAccess::Read),
                        })
                    },
                    MetalArgumentType::Constant((ty, MetalConstantType::Scalar)) => Some(KernelArgument {
                        name: a.name.clone(),
                        conditional: a.condition.is_some(),
                        ty: KernelArgumentType::Constant(ty.clone()),
                    }),
                    MetalArgumentType::Constant((ty, MetalConstantType::Array(None))) => Some(KernelArgument {
                        name: a.name.clone(),
                        conditional: a.condition.is_some(),
                        ty: KernelArgumentType::Constant(format!("&[{ty}]").into_boxed_str()),
                    }),
                    MetalArgumentType::Constant((ty, MetalConstantType::Array(Some(size)))) => Some(KernelArgument {
                        name: a.name.clone(),
                        conditional: a.condition.is_some(),
                        ty: KernelArgumentType::Constant(format!("&[{ty}; {size}]").into_boxed_str()),
                    }),
                    _ => None,
                })
                .collect(),
        })
    }
}

impl MetalKernelInfo {
    pub fn from_ast_node_and_source(
        node: MetalAstNode,
        source: &str,
        enum_paths: &EnumPaths,
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

        let mut arg_nodes = Vec::new();
        let mut annotations = Vec::new();

        for node in node.inner {
            match node.kind {
                MetalAstKind::ParmVarDecl {
                    name: _,
                    range: _,
                    ty: _,
                } => arg_nodes.push(node),
                MetalAstKind::AnnotateAttr => annotations.push(annotation_from_ast_node(node)?),
                _ => (),
            }
        }

        let annotations = annotations
            .into_iter()
            .map(|a| {
                if !a.is_empty() {
                    let mut a = a.into_vec();
                    Ok((a.remove(0), a.into_boxed_slice()))
                } else {
                    bail!("zero length annotation");
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        if !annotations.iter().any(|(k, _)| k.as_ref() == "dsl.kernel") {
            return Ok(None);
        }

        let public = annotations.iter().any(|(k, _)| k.as_ref() == "dsl.public");

        let declared: Vec<(Box<str>, Box<[Box<str>]>)> = annotations
            .iter()
            .filter(|(k, _)| k.as_ref() == "dsl.variants")
            .map(|(_, v)| {
                let [variant_name, variant_values] = v.as_ref() else {
                    bail!("malformed dsl.variants annotation");
                };

                let variant_values = variant_values
                    .split(',')
                    .map(|v| v.trim())
                    .filter(|v| !v.is_empty())
                    .map(Box::<str>::from)
                    .collect::<Box<[Box<str>]>>();

                Ok((variant_name.clone(), variant_values))
            })
            .collect::<anyhow::Result<_>>()?;

        if !is_template && !declared.is_empty() {
            bail!("VARIANTS annotation on a kernel that is not a template");
        }

        let variants = if is_template {
            let template_names = template_parameters.iter().map(|(name, _)| name.as_ref()).collect::<Vec<_>>();
            let declared_names = declared.iter().map(|(name, _)| name.as_ref()).collect::<Vec<_>>();

            if let Some(unknown) = declared_names.iter().find(|name| !template_names.contains(name)) {
                bail!("VARIANTS({unknown}, ...) does not name a template parameter; declared: {template_names:?}");
            }

            // Annotations may cover only some parameters now, but the ones present must
            // still read in declaration order — that ordering is what makes a VARIANTS
            // list next to its parameter obviously the list for that parameter.
            let expected_order =
                template_names.iter().copied().filter(|name| declared_names.contains(name)).collect::<Vec<_>>();
            if declared_names != expected_order {
                bail!("VARIANTS annotations {declared_names:?} are not in template parameter order {expected_order:?}");
            }

            let mut declared: HashMap<Box<str>, Box<[Box<str>]>> = declared.into_iter().collect();

            Some(
                template_parameters
                    .into_iter()
                    .map(|(name, ty)| {
                        let ty = match ty {
                            None => MetalTemplateParameterType::Type,
                            Some(ntt) => MetalTemplateParameterType::Value(MetalArgument::scalar_type_to_rust(
                                ntt.desugared_qual_type.unwrap_or(ntt.qual_type).as_ref(),
                            )?),
                        };

                        let variants = match declared.remove(&name).filter(|values| !values.is_empty()) {
                            Some(values) => values,
                            None => default_variants(&name, &ty, enum_paths)?,
                        };

                        Ok(MetalTemplateParameter {
                            name,
                            ty,
                            variants,
                        })
                    })
                    .collect::<anyhow::Result<_>>()?,
            )
        } else {
            None
        };

        let constraints: Box<[_]> = annotations
            .iter()
            .filter(|(k, _)| k.as_ref() == "dsl.constraint")
            .map(|(_, v)| {
                let [constraint_expr] = v.as_ref() else {
                    bail!("malformed dsl.constraint annotation");
                };

                Ok(constraint_expr.clone())
            })
            .collect::<anyhow::Result<_>>()?;

        let integer_defines = integer_object_defines_from_source(source);
        let arguments = arg_nodes
            .into_iter()
            .map(|an| MetalArgument::from_ast_node_and_source(an, source, &integer_defines))
            .collect::<anyhow::Result<Box<[_]>>>()?;

        Ok(Some(MetalKernelInfo {
            public,
            name: KernelName::from(name),
            arguments,
            variants,
            constraints,
        }))
    }
}
