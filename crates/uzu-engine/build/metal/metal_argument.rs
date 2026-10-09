use anyhow::{Context, bail};

use super::{
    IntegerObjectDefines, MetalAstType, annotation_from_ast_node,
    ast::{MetalArgumentType, MetalAstKind, MetalAstNode},
    parse_argument_annotation,
};
use crate::common::{KernelParameterType, identifiers::ArgumentName, kernel::KernelParameter};

#[derive(Debug)]
pub struct MetalArgument {
    pub name: ArgumentName,
    pub c_type: Box<str>,
    pub argument_type: MetalArgumentType,
    pub condition: Option<Box<str>>,
}

impl MetalArgument {
    pub fn scalar_type_to_rust(c_type: &str) -> anyhow::Result<Box<str>> {
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

    pub fn from_ast_node_and_source(
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
        })
    }

    pub fn to_parameter(&self) -> Option<KernelParameter> {
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
