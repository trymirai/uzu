use std::{path::Path, process::Command};

use serde::Deserialize;
use tempfile::NamedTempFile;

#[derive(PartialEq, Clone, Debug)]
pub enum CompiledArgType {
    Buffer,
    Constant(&'static str),
}

#[derive(Clone, Debug)]
pub struct Compiled {
    pub name: Box<str>,
    pub args: Box<[(Box<str>, CompiledArgType)]>,
    pub global_size: [Box<str>; 3],
    pub local_size: [Box<str>; 3],
    pub dependencies: Box<[Box<str>]>,
    pub mtlb: Box<[u8]>,
}

fn between<'a>(
    x: &'a str,
    s: &str,
    e: &str,
) -> &'a str {
    assert_eq!(x.match_indices(s).count(), 1);
    assert_eq!(x.match_indices(e).count(), 1);

    &x[x.find(s).unwrap() + s.len()..x.find(e).unwrap()]
}

type Node = clang_ast::Node<Clang>;

#[derive(Debug, Deserialize)]
struct TypeInfo {
    #[serde(rename = "qualType")]
    qual_type: Box<str>,
}

#[derive(Debug, Deserialize)]
enum Clang {
    ParmVarDecl {
        name: Box<str>,
        #[serde(rename = "type")]
        ty: TypeInfo,
    },
    Other,
}

fn args_from_json(json: &str) -> Box<[(Box<str>, CompiledArgType)]> {
    serde_json::from_str::<Node>(json)
        .unwrap()
        .inner
        .into_iter()
        .filter_map(|n| {
            if let Clang::ParmVarDecl {
                name,
                ty: TypeInfo {
                    qual_type,
                },
            } = n.kind
            {
                let mut words = qual_type.as_ref().split_whitespace().rev();

                match words.next() {
                    Some("*") => Some((name, CompiledArgType::Buffer)),
                    Some("&") => Some((
                        name,
                        CompiledArgType::Constant(match words.next() {
                            Some("uint") => "u32",
                            Some("float") => "f32",
                            _ => panic!("{}", qual_type.as_ref()),
                        }),
                    )),
                    _ if n.inner.len() > 0 => None, // global/local sizes
                    _ => panic!("{}", qual_type.as_ref()),
                }
            } else {
                None
            }
        })
        .collect()
}

pub fn compile(src: &Path) -> Compiled {
    let mut cmd = Command::new("xcrun");
    cmd.args(&[
        "-sdk",
        "macosx",
        "metal",
        "-x",
        "metal",
        "-std=metal3.2",
        "-MM",
        src.to_str().unwrap(),
        "-o",
        "-",
    ]);
    let stdout = cmd.output().unwrap().stdout;
    let dependencies = depfile::parse(str::from_utf8(&stdout).unwrap())
        .unwrap()
        .iter()
        .flat_map(|(_a, b)| b)
        .map(|x| Box::from(x.as_ref()))
        .collect();

    let mut cmd = Command::new("xcrun");
    cmd.args(&[
        "-sdk",
        "macosx",
        "metal",
        "-x",
        "metal",
        "-std=metal3.2",
        "-DSPECIALIZE(T, ...)=__DSL_SPECIALIZE_START__ T __DSL_SPECIALIZE_SEPARATOR__ __VA_ARGS__ __DSL_SPECIALIZE_END__",
        "-DKERNEL(NAME)=__DSL_KERNEL_START__ NAME __DSL_KERNEL_END__",
        "-DGET_MACRO(_1, _2, _3, NAME, ...)=NAME",
        "-DGLOBAL1(X)=__DSL_GLOBAL_START__ X __DSL_GLOBAL_SEPARATOR__ 1 __DSL_GLOBAL_SEPARATOR__ 1 __DSL_GLOBAL_END__",
        "-DGLOBAL2(X, Y)=__DSL_GLOBAL_START__ X __DSL_GLOBAL_SEPARATOR__ Y __DSL_GLOBAL_SEPARATOR__ 1 __DSL_GLOBAL_END__",
        "-DGLOBAL3(X, Y, Z)=__DSL_GLOBAL_START__ X __DSL_GLOBAL_SEPARATOR__ Y __DSL_GLOBAL_SEPARATOR__ Z __DSL_GLOBAL_END__",
        "-DGLOBAL(...)=GET_MACRO(__VA_ARGS__, GLOBAL3, GLOBAL2, GLOBAL1)(__VA_ARGS__)",
        "-DLOCAL1(X)=__DSL_LOCAL_START__ X __DSL_LOCAL_SEPARATOR__ 1 __DSL_LOCAL_SEPARATOR__ 1 __DSL_LOCAL_END__",
        "-DLOCAL2(X, Y)=__DSL_LOCAL_START__ X __DSL_LOCAL_SEPARATOR__ Y __DSL_LOCAL_SEPARATOR__ 1 __DSL_LOCAL_END__",
        "-DLOCAL3(X, Y, Z)=__DSL_LOCAL_START__ X __DSL_LOCAL_SEPARATOR__ Y __DSL_LOCAL_SEPARATOR__ Z __DSL_LOCAL_END__",
        "-DLOCAL(...)=GET_MACRO(__VA_ARGS__, LOCAL3, LOCAL2, LOCAL1)(__VA_ARGS__)",
        "-E",
        "-P",
        src.to_str().unwrap(),
        "-o",
        "-",
    ]);
    let stdout = cmd.output().unwrap().stdout;
    let preprocessed = str::from_utf8(&stdout).unwrap();

    let specialization_key = between(
        preprocessed,
        "__DSL_SPECIALIZE_START__ ",
        " __DSL_SPECIALIZE_SEPARATOR__",
    );

    let specialization_values: Box<[&str]> = between(
        preprocessed,
        "__DSL_SPECIALIZE_SEPARATOR__ ",
        " __DSL_SPECIALIZE_END__",
    )
    .split(",")
    .map(|x| x.trim())
    .collect();

    let name =
        between(preprocessed, "__DSL_KERNEL_START__ ", " __DSL_KERNEL_END__");

    let global_size: [Box<str>; 3] =
        between(preprocessed, "__DSL_GLOBAL_START__ ", " __DSL_GLOBAL_END__")
            .split(" __DSL_GLOBAL_SEPARATOR__ ")
            .map(|x| Box::from(x.trim()))
            .collect::<Vec<Box<str>>>()
            .try_into()
            .unwrap();

    let local_size: [Box<str>; 3] =
        between(preprocessed, "__DSL_LOCAL_START__ ", " __DSL_LOCAL_END__")
            .split(" __DSL_LOCAL_SEPARATOR__ ")
            .map(|x| Box::from(x.trim()))
            .collect::<Vec<Box<str>>>()
            .try_into()
            .unwrap();

    let max_total_threads_per_threadgroup = local_size.join(" * ");

    let mut cmd = Command::new("xcrun");
    cmd.args(&[
            "-sdk",
            "macosx",
            "metal",
            "-x",
            "metal",
            "-std=metal3.2",
            &format!("-DSPECIALIZE(T, ...)=typedef {} {};", specialization_values.first().unwrap(), specialization_key),
            &format!(
                "-DKERNEL(NAME)=[[kernel]] [[max_total_threads_per_threadgroup({})]] void {}_{}",
                max_total_threads_per_threadgroup,
                name,
                specialization_values.first().unwrap(),
            ),
            "-DGLOBAL(...)=[[threadgroup_position_in_grid]]",
            "-DLOCAL(...)=[[thread_position_in_threadgroup]]",
            "-fsyntax-only",
            "-Xclang",
            "-ast-dump=json",
            "-Xclang",
            &format!("-ast-dump-filter={}_{}", name, specialization_values.first().unwrap()),
            src.to_str().unwrap(),
            "-o",
            "-",
        ]);

    let args =
        args_from_json(str::from_utf8(&cmd.output().unwrap().stdout).unwrap());

    let airs = specialization_values.iter().map(|specialization_value| {
        let air_file = NamedTempFile::new().unwrap();
        let mut cmd = Command::new("xcrun");
        cmd.args(&[
            "-sdk",
            "macosx",
            "metal",
            "-x",
            "metal",
            "-std=metal3.2",
            &format!("-DSPECIALIZE(T, ...)=typedef {specialization_value} {specialization_key};"),
            &format!(
                "-DKERNEL(NAME)=[[kernel]] [[max_total_threads_per_threadgroup({max_total_threads_per_threadgroup})]] void {name}_{specialization_value}"
            ),
            "-DGLOBAL(...)=[[threadgroup_position_in_grid]]",
            "-DLOCAL(...)=[[thread_position_in_threadgroup]]",
            "-O2",
            "-c",
            src.to_str().unwrap(),
            "-o",
            air_file.path().to_str().unwrap(),
        ]);

        cmd.status().expect("failed to compile to air");

        air_file
    }).collect::<Box<[NamedTempFile]>>();

    let mut cmd = Command::new("xcrun");
    cmd.args(&["-sdk", "macosx", "metallib"]);
    cmd.args(
        &airs
            .iter()
            .map(|x| x.path().to_str().unwrap())
            .collect::<Box<[&str]>>(),
    );
    cmd.args(&["-o", "-"]);
    let mtlb = cmd.output().unwrap().stdout.into_boxed_slice();

    Compiled {
        name: Box::from(name),
        args,
        global_size,
        local_size,
        dependencies,
        mtlb,
    }
}
