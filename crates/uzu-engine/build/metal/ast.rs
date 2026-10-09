use anyhow::{Context, bail};
use quote::quote;
use serde::Deserialize;

use super::{IntegerObjectDefines, MetalArgument, MetalAstType};
use crate::common::expr_rewrite::rewrite_paths_with;

pub type MetalAstNode = clang_ast::Node<MetalAstKind>;

pub fn integer_object_defines_from_source(source: &str) -> IntegerObjectDefines {
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

pub fn annotation_from_ast_node(annotation_node: MetalAstNode) -> anyhow::Result<Box<[Box<str>]>> {
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

#[derive(Debug, Clone, Copy)]
pub enum MetalBufferAccess {
    Read,
    ReadWrite,
}

#[derive(Debug, Clone)]
pub enum MetalConstantType {
    Scalar,
    Array(Option<Box<str>>),
}

#[derive(Debug, Clone)]
pub enum MetalGroupsType {
    Direct(Box<str>),
    Indirect,
}

#[derive(Debug, Clone)]
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

pub fn parse_argument_annotation(
    c_type: &str,
    source: &str,
    annotation: Option<&[Box<str>]>,
    integer_defines: &IntegerObjectDefines,
) -> anyhow::Result<(MetalArgumentType, Option<Box<str>>)> {
    if let Some(annotation) = annotation
        && annotation.first().map(|s| s.as_ref()) == Some("dsl.specialize_if")
    {
        if annotation.len() != 2 {
            bail!("dsl.specialize_if takes 1 argument, found {}", annotation.len() - 1);
        }
        let ty = MetalArgument::scalar_type_to_rust(c_type)?;
        return Ok((MetalArgumentType::Specialize(format!("Option < {ty} >").into()), Some(annotation[1].clone())));
    }

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

#[derive(Debug)]
pub enum MetalTemplateParameterType {
    Type,
    Value(Box<str>),
}
