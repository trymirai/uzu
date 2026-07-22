//! Type checker and interpreter for shader `CONSTRAINT(...)` expressions.
//!
//! A constraint prunes a kernel's declared variant cross-product down to the set that
//! actually gets compiled, so a constraint that silently evaluates the wrong way
//! over- or under-instantiates the shipped kernel set with no diagnostic. This module
//! makes that class of mistake a build error: every identifier must resolve to an axis
//! of the kernel being constrained, and every literal compared for equality against an
//! axis must be a member of that axis's declared value set. `BT != "flaot"` and
//! `BITS == 3` stop building instead of quietly changing what ships.
//!
//! The grammar is a subset of Rust expression syntax, so `syn` does the parsing.
//! Anything outside the whitelist below is an error, not a feature request.

use std::{
    collections::BTreeMap,
    fmt::{self, Display},
};

use anyhow::{Context, bail};
use itertools::Itertools;
use syn::{BinOp, Expr, Lit, UnOp};

/// The type of an axis, or of a resolved sub-expression.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Type {
    Bool,
    Int,
    /// A `typename` template parameter, whose values are Metal type names.
    DType,
    /// A gpu_types enum, held by short name (`GemmTiling`).
    Enum(Box<str>),
}

impl Display for Type {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Type::Bool => f.write_str("bool"),
            Type::Int => f.write_str("integer"),
            Type::DType => f.write_str("data type"),
            Type::Enum(name) => write!(f, "{name}"),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Value {
    Bool(bool),
    Int(i64),
    DType(Box<str>),
    EnumVariant {
        enum_name: Box<str>,
        variant: Box<str>,
    },
}

impl Value {
    pub fn ty(&self) -> Type {
        match self {
            Value::Bool(_) => Type::Bool,
            Value::Int(_) => Type::Int,
            Value::DType(_) => Type::DType,
            Value::EnumVariant {
                enum_name,
                ..
            } => Type::Enum(enum_name.clone()),
        }
    }
}

impl Display for Value {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{b}"),
            Value::Int(i) => write!(f, "{i}"),
            Value::DType(name) => write!(f, "{name}"),
            Value::EnumVariant {
                enum_name,
                variant,
            } => write!(f, "{enum_name}::{variant}"),
        }
    }
}

/// One template axis: its name, its type, and the exact set of values the shader
/// declared for it via `VARIANTS`.
#[derive(Clone, Debug)]
pub struct AxisDecl {
    pub name: Box<str>,
    pub ty: Type,
    pub values: Box<[Value]>,
}

impl AxisDecl {
    /// Parses a `VARIANTS` entry as written in the shader against this axis's type.
    pub fn parse_value(
        ty: &Type,
        text: &str,
    ) -> anyhow::Result<Value> {
        let text = text.trim();
        Ok(match ty {
            Type::Bool => match text {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                _ => bail!("expected `true` or `false`, found `{text}`"),
            },
            Type::Int => Value::Int(text.parse().with_context(|| format!("`{text}` is not an integer"))?),
            Type::DType => Value::DType(text.into()),
            Type::Enum(enum_name) => {
                let Some((namespace, variant)) = text.rsplit_once("::") else {
                    bail!("expected `{enum_name}::<variant>`, found `{text}`");
                };
                if namespace != enum_name.as_ref() {
                    bail!("expected a `{enum_name}` variant, found `{text}`");
                }
                Value::EnumVariant {
                    enum_name: enum_name.clone(),
                    variant: variant.into(),
                }
            },
        })
    }
}

/// A constraint after resolution: every leaf carries the type it was checked at, so the
/// interpreter never has to re-resolve a name. Both backends walk this one tree, which
/// is what keeps build-time pruning and any generated runtime check from disagreeing.
#[derive(Clone, Debug)]
pub enum ResolvedExpr {
    Axis {
        name: Box<str>,
        ty: Type,
    },
    Literal(Value),
    Not(Box<ResolvedExpr>),
    Binary {
        op: Op,
        lhs: Box<ResolvedExpr>,
        rhs: Box<ResolvedExpr>,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Op {
    Eq,
    Ne,
    Lt,
    Le,
    And,
    Or,
}

impl Op {
    fn result_type(&self) -> Type {
        Type::Bool
    }

    fn is_ordering(&self) -> bool {
        matches!(self, Op::Lt | Op::Le)
    }
}

/// A kernel's axes plus its type-checked constraints.
pub struct ConstraintSet {
    axes: BTreeMap<Box<str>, AxisDecl>,
    constraints: Box<[Constraint]>,
}

struct Constraint {
    /// The expression as written in the shader, for diagnostics.
    source: Box<str>,
    expr: ResolvedExpr,
}

impl ConstraintSet {
    /// Type-checks `constraints` against `axes`. `kernel` only appears in error messages.
    pub fn compile(
        kernel: &str,
        axes: impl IntoIterator<Item = AxisDecl>,
        constraints: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> anyhow::Result<Self> {
        let axes: BTreeMap<Box<str>, AxisDecl> = axes.into_iter().map(|a| (a.name.clone(), a)).collect();

        let constraints = constraints
            .into_iter()
            .map(|source| {
                let source = source.as_ref();
                let expr = compile_expr(&axes, source)
                    .with_context(|| format!("in CONSTRAINT({source}) of kernel `{kernel}`"))?;
                Ok(Constraint {
                    source: source.into(),
                    expr,
                })
            })
            .collect::<anyhow::Result<Box<[_]>>>()?;

        Ok(Self {
            axes,
            constraints,
        })
    }

    /// Whether every constraint holds for this assignment of axis values. Bindings are
    /// the shader-spelled variant strings, parsed against each axis's declared type.
    pub fn satisfied<N: AsRef<str>, V: AsRef<str>>(
        &self,
        bindings: &[(N, V)],
    ) -> anyhow::Result<bool> {
        if self.constraints.is_empty() {
            return Ok(true);
        }

        let bindings = bindings
            .iter()
            .map(|(name, text)| {
                let name = name.as_ref();
                let axis = self.axes.get(name).with_context(|| format!("no axis named `{name}`"))?;
                let value = AxisDecl::parse_value(&axis.ty, text.as_ref())
                    .with_context(|| format!("binding for axis `{name}`"))?;
                Ok((name.to_owned(), value))
            })
            .collect::<anyhow::Result<BTreeMap<String, Value>>>()?;

        for constraint in self.constraints.iter() {
            let holds = eval(&constraint.expr, &bindings)
                .with_context(|| format!("evaluating CONSTRAINT({})", constraint.source))?;
            if !holds {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

fn compile_expr(
    axes: &BTreeMap<Box<str>, AxisDecl>,
    source: &str,
) -> anyhow::Result<ResolvedExpr> {
    let expr: Expr = syn::parse_str(source).with_context(|| format!("cannot parse `{source}`"))?;
    let resolved = resolve(axes, &expr)?;

    let ty = type_of(&resolved);
    if ty != Type::Bool {
        bail!("a constraint must be a bool expression, this one is {ty}");
    }

    Ok(resolved)
}

fn type_of(expr: &ResolvedExpr) -> Type {
    match expr {
        ResolvedExpr::Axis {
            ty,
            ..
        } => ty.clone(),
        ResolvedExpr::Literal(value) => value.ty(),
        ResolvedExpr::Not(_) => Type::Bool,
        ResolvedExpr::Binary {
            op,
            ..
        } => op.result_type(),
    }
}

fn resolve(
    axes: &BTreeMap<Box<str>, AxisDecl>,
    expr: &Expr,
) -> anyhow::Result<ResolvedExpr> {
    match expr {
        Expr::Paren(paren) => resolve(axes, &paren.expr),

        Expr::Unary(unary) => {
            let UnOp::Not(_) = unary.op else {
                bail!("only `!` is allowed as a unary operator");
            };
            let operand = resolve(axes, &unary.expr)?;
            let ty = type_of(&operand);
            if ty != Type::Bool {
                bail!("`!` needs a bool operand, found {ty}");
            }
            Ok(ResolvedExpr::Not(Box::new(operand)))
        },

        Expr::Binary(binary) => {
            let op = match binary.op {
                BinOp::Eq(_) => Op::Eq,
                BinOp::Ne(_) => Op::Ne,
                BinOp::Lt(_) => Op::Lt,
                BinOp::Le(_) => Op::Le,
                BinOp::And(_) => Op::And,
                BinOp::Or(_) => Op::Or,
                _ => bail!("operator `{}` is not allowed in a constraint", quote::quote!(#binary.op)),
            };

            let lhs = resolve(axes, &binary.left)?;
            let rhs = resolve(axes, &binary.right)?;
            let (lhs_type, rhs_type) = (type_of(&lhs), type_of(&rhs));

            if lhs_type != rhs_type {
                bail!("cannot compare {lhs_type} with {rhs_type}");
            }

            match op {
                Op::And | Op::Or if lhs_type != Type::Bool => {
                    bail!("`&&` and `||` need bool operands, found {lhs_type}")
                },
                _ if op.is_ordering() && lhs_type != Type::Int => {
                    bail!("`<` and `<=` need integer operands, found {lhs_type}")
                },
                _ => (),
            }

            // The typo killer: `BITS == 3` or `BT != "flaot"` compares an axis against a
            // value it can never hold, so the comparison is a constant. Ordering
            // comparisons are exempt — a bound need not itself be a declared value.
            if matches!(op, Op::Eq | Op::Ne) {
                check_membership(axes, &lhs, &rhs)?;
                check_membership(axes, &rhs, &lhs)?;
            }

            Ok(ResolvedExpr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            })
        },

        Expr::Path(path) => resolve_path(axes, &path.path),

        Expr::Lit(lit) => Ok(ResolvedExpr::Literal(match &lit.lit {
            Lit::Bool(b) => Value::Bool(b.value),
            Lit::Int(i) => Value::Int(i.base10_parse()?),
            Lit::Str(s) => Value::DType(s.value().into()),
            other => bail!("literal `{}` is not allowed in a constraint", quote::quote!(#other)),
        })),

        other => bail!("`{}` is not allowed in a constraint", quote::quote!(#other)),
    }
}

fn resolve_path(
    axes: &BTreeMap<Box<str>, AxisDecl>,
    path: &syn::Path,
) -> anyhow::Result<ResolvedExpr> {
    let segments = path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>();

    match segments.as_slice() {
        [name] => {
            let axis = axes.get(name.as_str()).with_context(|| {
                format!("`{name}` is not an axis of this kernel (declared: {})", axes.keys().join(", "))
            })?;
            Ok(ResolvedExpr::Axis {
                name: axis.name.clone(),
                ty: axis.ty.clone(),
            })
        },
        [enum_name, variant] => Ok(ResolvedExpr::Literal(Value::EnumVariant {
            enum_name: enum_name.as_str().into(),
            variant: variant.as_str().into(),
        })),
        _ => bail!("`{}` is not an axis name or an enum variant", segments.join("::")),
    }
}

/// If `subject` is an axis and `literal` is a constant, requires the constant to be one
/// of the axis's declared values.
fn check_membership(
    axes: &BTreeMap<Box<str>, AxisDecl>,
    subject: &ResolvedExpr,
    literal: &ResolvedExpr,
) -> anyhow::Result<()> {
    let (
        ResolvedExpr::Axis {
            name,
            ..
        },
        ResolvedExpr::Literal(value),
    ) = (subject, literal)
    else {
        return Ok(());
    };

    let axis = &axes[name];
    if !axis.values.contains(value) {
        bail!(
            "`{value}` is not a declared value of axis `{name}` (declared: {})",
            axis.values.iter().map(|v| v.to_string()).join(", ")
        );
    }

    Ok(())
}

fn eval(
    expr: &ResolvedExpr,
    bindings: &BTreeMap<String, Value>,
) -> anyhow::Result<bool> {
    Ok(match value_of(expr, bindings)? {
        Value::Bool(b) => b,
        other => bail!("expected a bool, found `{other}`"),
    })
}

fn value_of(
    expr: &ResolvedExpr,
    bindings: &BTreeMap<String, Value>,
) -> anyhow::Result<Value> {
    Ok(match expr {
        ResolvedExpr::Axis {
            name,
            ..
        } => bindings.get(name.as_ref()).with_context(|| format!("axis `{name}` is unbound"))?.clone(),

        ResolvedExpr::Literal(value) => value.clone(),

        ResolvedExpr::Not(operand) => Value::Bool(!eval(operand, bindings)?),

        ResolvedExpr::Binary {
            op,
            lhs,
            rhs,
            ..
        } => Value::Bool(match op {
            Op::And => eval(lhs, bindings)? && eval(rhs, bindings)?,
            Op::Or => eval(lhs, bindings)? || eval(rhs, bindings)?,
            Op::Eq => value_of(lhs, bindings)? == value_of(rhs, bindings)?,
            Op::Ne => value_of(lhs, bindings)? != value_of(rhs, bindings)?,
            Op::Lt | Op::Le => {
                let (Value::Int(l), Value::Int(r)) = (value_of(lhs, bindings)?, value_of(rhs, bindings)?) else {
                    bail!("ordering comparison on non-integers");
                };
                if matches!(op, Op::Lt) {
                    l < r
                } else {
                    l <= r
                }
            },
        }),
    })
}
