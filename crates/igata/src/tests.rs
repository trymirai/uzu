//! The point of the crate: the variant-space logic is a build script everywhere else,
//! so nothing here would otherwise be executable by `cargo test`.

use crate::{
    constraint_expr::Type,
    enum_paths::EnumPaths,
    gpu_types::{GpuTypeFile, GpuTypes, tile_geometry::TileGeometry},
    mangling::{snake_case, static_mangle},
    variants::{AxisSpec, KernelSpace},
};

/// The shape of the real `gemm` gpu types, cut down to what the tests need.
const GPU_TYPES: &str = r#"
    #[repr(C)]
    pub enum GemmTiling {
        Tile8x32x32_Simdgroups1x1,
        Tile64x64x256_Simdgroups2x2,
    }

    #[repr(C)]
    pub enum GemmBPrologueKind {
        FullPrecision,
        ScaleBiasDequant,
        ScaleZeroPointDequant,
        ScaleSymmetricDequant,
    }

    #[repr(C)]
    pub enum QuantPrologue {
        ScaleBiasDequant = 1,
        ScaleZeroPointDequant = 2,
        ScaleSymmetricDequant = 3,
    }

    #[repr(C)]
    pub enum QuantBits { B4 = 4, B8 = 8 }

    #[repr(C)]
    pub enum QuantGroupSize { G16 = 16, G32 = 32, G64 = 64, G128 = 128 }

    #[variant_group(B_PROLOGUE, BITS, GROUP_SIZE)]
    pub enum WeightsKey {
        FullPrecision,
        Quant {
            b_prologue: QuantPrologue,
            bits: QuantBits,
            group_size: QuantGroupSize,
        },
    }
"#;

fn parse(source: &str) -> anyhow::Result<EnumPaths> {
    EnumPaths::from_gpu_types(&GpuTypes::new([GpuTypeFile::parse("gemm", source)?]))
}

fn enum_paths() -> EnumPaths {
    parse(GPU_TYPES).unwrap()
}

fn axis(
    name: &str,
    ty: Type,
    values: &[&str],
) -> AxisSpec {
    AxisSpec {
        name: name.into(),
        ty,
        values: values.iter().map(|value| (*value).into()).collect(),
    }
}

/// The three axes `WeightsKey` groups, declared as the shaders declare them.
fn weights_axes() -> Vec<AxisSpec> {
    vec![
        axis(
            "B_PROLOGUE",
            Type::Enum("GemmBPrologueKind".into()),
            &[
                "GemmBPrologueKind::FullPrecision",
                "GemmBPrologueKind::ScaleBiasDequant",
                "GemmBPrologueKind::ScaleZeroPointDequant",
                "GemmBPrologueKind::ScaleSymmetricDequant",
            ],
        ),
        axis("BITS", Type::Int, &["0", "4", "8"]),
        axis("GROUP_SIZE", Type::Int, &["0", "16", "32", "64", "128"]),
    ]
}

fn space<'a>(
    axes: &'a [AxisSpec],
    constraints: &'a [Box<str>],
) -> KernelSpace<'a> {
    KernelSpace {
        name: "Kernel",
        axes: Some(axes),
        constraints,
    }
}

fn constraints(sources: &[&str]) -> Vec<Box<str>> {
    sources.iter().map(|source| (*source).into()).collect()
}

/// The message of an error and everything it was raised from.
fn error(result: anyhow::Result<impl Sized>) -> String {
    format!("{:#}", result.err().expect("expected an error"))
}

fn compile_error(
    axes: &[AxisSpec],
    source: &str,
) -> String {
    let constraints = constraints(&[source]);
    error(space(axes, &constraints).constraint_set(&enum_paths()))
}

fn accepted(
    axes: &[AxisSpec],
    sources: &[&str],
) -> usize {
    let constraints = constraints(sources);
    space(axes, &constraints).accepted_variants(&enum_paths()).unwrap().len()
}

// -- constraints ------------------------------------------------------------------

#[test]
fn a_misspelled_dtype_is_not_a_silently_false_comparison() {
    let axes = vec![axis("BT", Type::DType, &["bfloat", "float"])];
    let message = compile_error(&axes, r#"BT != "flaot""#);
    assert!(message.contains("`flaot` is not a declared value of axis `BT`"), "{message}");
    assert!(message.contains("(declared: bfloat, float)"), "{message}");
}

#[test]
fn an_undeclared_integer_is_not_a_silently_false_comparison() {
    let axes = weights_axes();
    let message = compile_error(&axes, "BITS == 3");
    assert!(message.contains("`3` is not a declared value of axis `BITS`"), "{message}");
    assert!(message.contains("(declared: 0, 4, 8)"), "{message}");
}

#[test]
fn an_unknown_name_lists_the_axes_that_do_exist() {
    let axes = weights_axes();
    let message = compile_error(&axes, "BIT == 4");
    assert!(message.contains("`BIT` is not an axis of this kernel"), "{message}");
    assert!(message.contains("(declared: BITS, B_PROLOGUE, GROUP_SIZE)"), "{message}");
}

#[test]
fn types_may_not_be_mixed() {
    let axes = weights_axes();
    assert!(compile_error(&axes, "BITS == true").contains("cannot compare integer with bool"));
    assert!(compile_error(&axes, "BITS && BITS").contains("`&&` and `||` need bool operands, found integer"));
    assert!(compile_error(&axes, "BITS").contains("a constraint must be a bool expression, this one is integer"));
}

#[test]
fn ordering_needs_integers() {
    let axes = vec![axis("BT", Type::DType, &["bfloat", "float"])];
    assert!(compile_error(&axes, r#"BT < "float""#).contains("`<` and `<=` need integer operands"));
}

#[test]
fn only_generated_helpers_may_be_called() {
    let axes = vec![axis("GEMM_TILING", Type::Enum("GemmTiling".into()), &["GemmTiling::Tile8x32x32_Simdgroups1x1"])];
    let message = compile_error(&axes, "block_m(GEMM_TILING) == 8");
    assert!(message.contains("`block_m` is not a generated helper"), "{message}");
    assert!(message.contains("gemm_tiling_block_m"), "{message}");

    assert!(compile_error(&axes, "gemm_tiling_block_m() == 8").contains("takes exactly one argument"));
    assert!(compile_error(&axes, "gemm_tiling_block_m(8) == 8").contains("`gemm_tiling_block_m` takes GemmTiling"));

    let message = compile_error(&axes, "gemm_tiling_block_m(GEMM_TILING) == 7");
    assert!(message.contains("`gemm_tiling_block_m` never returns `7`"), "{message}");
    assert!(message.contains("(it returns one of: 8, 64)"), "{message}");
}

#[test]
fn the_grammar_is_a_whitelist() {
    let axes = weights_axes();
    assert!(compile_error(&axes, "BITS.count() == 1").contains("is not allowed in a constraint"));
    assert!(compile_error(&axes, "BITS + 1 == 5").contains("`+` is not allowed"));
    assert!(compile_error(&axes, "-BITS == 4").contains("only `!` is allowed as a unary operator"));
    assert!(compile_error(&axes, "BITS ==").contains("cannot parse"));
}

#[test]
fn operators_evaluate_as_written() {
    let axes = vec![axis("A", Type::Int, &["0", "1"]), axis("B", Type::Int, &["0", "1"])];

    // Of the four (A, B) assignments, how many survive.
    assert_eq!(accepted(&axes, &["A == B"]), 2);
    assert_eq!(accepted(&axes, &["A != B"]), 2);
    assert_eq!(accepted(&axes, &["A < B"]), 1);
    assert_eq!(accepted(&axes, &["A <= B"]), 3);
    assert_eq!(accepted(&axes, &["A == 1 && B == 1"]), 1);
    assert_eq!(accepted(&axes, &["A == 1 || B == 1"]), 3);
    assert_eq!(accepted(&axes, &["!(A == 1 || B == 1)"]), 1);
    assert_eq!(accepted(&axes, &["A == 1", "B == 1"]), 1);
    assert_eq!(accepted(&axes, &[]), 4);
}

// -- variant groups ---------------------------------------------------------------

fn group_error(group: &str) -> String {
    error(parse(&format!("{GPU_TYPES}\n{group}")))
}

#[test]
fn a_field_must_be_named_after_its_axis() {
    let message = group_error(
        "#[variant_group(BITS, GROUP_SIZE)]
         pub enum Other { Quant { bits: QuantBits, size: QuantGroupSize } }",
    );
    assert!(message.contains("has no field `group_size` for axis `GROUP_SIZE`"), "{message}");
    assert!(message.contains("(its fields are: bits, size)"), "{message}");
}

#[test]
fn a_field_naming_no_axis_is_rejected() {
    let message = group_error(
        "#[variant_group(BITS)]
         pub enum Other { Quant { bits: QuantBits, group_size: QuantGroupSize } }",
    );
    assert!(message.contains("has field(s) `group_size` naming no axis"), "{message}");
    assert!(message.contains("(it groups: BITS)"), "{message}");
}

#[test]
fn only_one_arm_may_stand_for_the_leftover_values() {
    let message = group_error(
        "#[variant_group(BITS)]
         pub enum Other { None, Absent, Quant { bits: QuantBits } }",
    );
    assert!(message.contains("`Other` has more than one unit variant (`None` and `Absent`)"), "{message}");
}

#[test]
fn arms_must_name_their_fields() {
    let message = group_error(
        "#[variant_group(BITS)]
         pub enum Other { Quant(QuantBits) }",
    );
    assert!(message.contains("variant `Quant` must use named fields"), "{message}");
}

#[test]
fn a_member_must_match_a_declared_axis_value() {
    let mut axes = weights_axes();
    axes[1] = axis("BITS", Type::Int, &["0", "4"]);
    let message = error(space(&axes, &[]).accepted_variants(&enum_paths()));
    assert!(message.contains("`QuantBits::B8` does not match any declared value of axis `BITS`"), "{message}");
}

#[test]
fn the_unit_arm_needs_exactly_one_leftover_value() {
    let mut axes = weights_axes();
    axes[1] = axis("BITS", Type::Int, &["0", "2", "4", "8"]);
    let message = error(space(&axes, &[]).accepted_variants(&enum_paths()));
    assert!(message.contains("axis `BITS` must have exactly one value left over"), "{message}");
}

#[test]
fn two_groups_may_not_share_an_axis() {
    let paths = parse(&format!(
        "{GPU_TYPES}
         #[variant_group(BITS, GROUP_SIZE)]
         pub enum Other {{ Quant {{ bits: QuantBits, group_size: QuantGroupSize }} }}"
    ))
    .unwrap();
    let axes = weights_axes();
    let message = error(space(&axes, &[]).accepted_variants(&paths));
    assert!(message.contains("share an axis"), "{message}");
}

// -- enumeration ------------------------------------------------------------------

#[test]
fn a_variant_group_collapses_the_illegal_quadrant() {
    let axes = weights_axes();
    // 4 * 3 * 5 raw combinations; the sum type admits 3 * 2 * 4 quantized ones plus
    // full precision.
    assert_eq!(axes.iter().map(|axis| axis.values.len()).product::<usize>(), 60);
    assert_eq!(accepted(&axes, &[]), 25);
}

#[test]
fn constraints_prune_what_the_group_admits() {
    let axes = weights_axes();
    assert_eq!(accepted(&axes, &["BITS != 8"]), 13);
    assert_eq!(accepted(&axes, &["B_PROLOGUE == GemmBPrologueKind::FullPrecision"]), 1);
}

#[test]
fn a_kernel_with_no_axes_is_one_variant() {
    let space = KernelSpace {
        name: "Kernel",
        axes: None,
        constraints: &[],
    };
    assert_eq!(space.accepted_variants(&enum_paths()).unwrap(), vec![None]);
}

#[test]
fn accepted_variants_are_in_template_parameter_order() {
    let axes = weights_axes();
    let constraints = constraints(&["B_PROLOGUE == GemmBPrologueKind::FullPrecision"]);
    let variants = space(&axes, &constraints).accepted_variants(&enum_paths()).unwrap();
    assert_eq!(
        variants,
        vec![Some(vec![
            ("B_PROLOGUE".into(), "GemmBPrologueKind::FullPrecision".into()),
            ("BITS".into(), "0".into()),
            ("GROUP_SIZE".into(), "0".into()),
        ])]
    );
}

// -- tile geometry and mangling ---------------------------------------------------

#[test]
fn a_tile_name_is_its_geometry() {
    let geometry = TileGeometry::parse("Tile16x128x256_Simdgroups1x4").unwrap();
    assert_eq!(
        (geometry.block_m, geometry.block_n, geometry.block_k, geometry.simdgroups_m, geometry.simdgroups_n),
        (16, 128, 256, 1, 4)
    );
    assert!(TileGeometry::parse("Tile16x128_Simdgroups1x4").is_none());
    assert!(TileGeometry::parse("GemmTiling").is_none());
}

#[test]
fn one_malformed_variant_disqualifies_the_whole_enum() {
    let paths = parse(
        "#[repr(C)]
         pub enum Tilings { Tile8x32x32_Simdgroups1x1, Tile8x32_Simdgroups1x1 }",
    )
    .unwrap();
    assert!(paths.helpers().is_empty());
    assert!(!enum_paths().helpers().is_empty());
}

#[test]
fn names_are_mangled_the_way_the_wrappers_spell_them() {
    assert_eq!(static_mangle("Gemm", ["bfloat", "GemmTiling::Tile8x32x32_Simdgroups1x1", "0"]), {
        "_D4GemmS6VbfloatS25VTile8x32x32_Simdgroups1x1S1V0"
    });
    assert_eq!(static_mangle("Gemm", [] as [&str; 0]), "_D4Gemm");
    assert_eq!(static_mangle("Gemm", ["-1"]), "_D4GemmS2Vn1");
    assert_eq!(snake_case("WeightsKey"), "weights_key");
    assert_eq!(snake_case("Gemv"), "gemv");
}

#[test]
fn the_fingerprint_moves_only_when_the_definitions_do() {
    let unchanged = parse(&GPU_TYPES.replace("pub enum QuantBits", "pub  enum QuantBits")).unwrap();
    assert_eq!(enum_paths().semantic_fingerprint(), unchanged.semantic_fingerprint());

    let renamed = parse(&GPU_TYPES.replace("b_prologue", "prologue").replace("B_PROLOGUE", "PROLOGUE")).unwrap();
    assert_ne!(enum_paths().semantic_fingerprint(), renamed.semantic_fingerprint());
}
