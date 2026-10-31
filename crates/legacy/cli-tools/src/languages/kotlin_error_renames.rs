use std::{fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use heck::AsLowerCamelCase;

/// Applies each crate's `[bindings.kotlin.rename]` entries for error fields, like `"ChatSessionError.Backend.message"`.
///
/// Kotlin exceptions already have a `message`, so an error field with that name does not compile. UniFFI 0.31 reads
/// these renames but matches them against crate names while errors carry their full module path, so it skips every
/// error outside a crate root. UniFFI 0.32 applies them itself; this step goes once cargo-swift supports 0.32.
pub fn apply_kotlin_error_renames(
    crates_path: &Path,
    generated_sources_path: &Path,
) -> Result<()> {
    for entry in fs::read_dir(crates_path.join("legacy"))? {
        let config_path = entry?.path().join("uniffi.toml");
        if !config_path.is_file() {
            continue;
        }
        let config: toml::Table = toml::from_str(&fs::read_to_string(&config_path)?)?;
        let Some(kotlin) = config.get("bindings").and_then(|bindings| bindings.get("kotlin")) else {
            continue;
        };
        let Some(renames) = kotlin.get("rename").and_then(toml::Value::as_table) else {
            continue;
        };
        let package = kotlin
            .get("package_name")
            .and_then(toml::Value::as_str)
            .with_context(|| format!("{} renames Kotlin items but sets no package_name", config_path.display()))?;
        let package_path = generated_sources_path.join(package.replace('.', "/"));
        let source_path = fs::read_dir(&package_path)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .find(|path| path.extension().and_then(|extension| extension.to_str()) == Some("kt"))
            .with_context(|| format!("No Kotlin bindings in {}", package_path.display()))?;

        let mut source = fs::read_to_string(&source_path)?;
        for (key, new_name) in renames {
            let new_name = new_name.as_str().with_context(|| format!("Rename {key} is not a string"))?;
            let [error, variant, field] = key.split('.').collect::<Vec<_>>()[..] else {
                bail!("Rename {key} in {} is not an error field (Error.Variant.field)", config_path.display());
            };
            source = rename_error_field(&source, error, variant, field, new_name)
                .with_context(|| format!("Applying rename {key} from {}", config_path.display()))?;
        }
        fs::write(&source_path, source)?;
    }
    Ok(())
}

fn rename_error_field(
    source: &str,
    error: &str,
    variant: &str,
    field: &str,
    new_name: &str,
) -> Result<String> {
    let class_name = exception_name(error);
    let variant_name = exception_name(variant);
    let old_field = format!("`{}`", AsLowerCamelCase(field));
    let new_field = format!("`{}`", AsLowerCamelCase(new_name));

    // The variant's class: its field and the `message` getter that prints it.
    let error_class = source
        .find(&format!("sealed class {class_name}: kotlin.Exception()"))
        .ok_or_else(|| anyhow!("No exception class {class_name}"))?;
    let declaration_start = error_class
        + source[error_class..]
            .find(&format!("    class {variant_name}("))
            .ok_or_else(|| anyhow!("No variant {class_name}.{variant_name}"))?;
    let declaration_end = declaration_start
        + source[declaration_start..].find("\n    }\n").ok_or_else(|| anyhow!("Unterminated {variant_name}"))?;
    let declaration = &source[declaration_start..declaration_end];
    if !declaration.contains(&format!("val {old_field}:")) {
        bail!("{class_name}.{variant_name} has no field {old_field}");
    }
    let renamed = declaration
        .replace(&old_field, &new_field)
        .replace(&format!("{}=", AsLowerCamelCase(field)), &format!("{}=", AsLowerCamelCase(new_name)));
    let mut output = format!("{}{renamed}{}", &source[..declaration_start], &source[declaration_end..]);

    // The converter's `allocationSize` and `write` branches read the field by name.
    let branch_marker = format!("is {class_name}.{variant_name} -> ");
    let old_access = format!("value.{old_field}");
    let new_access = format!("value.{new_field}");
    let mut renamed_branches = 0;
    let mut search_from = 0;
    while let Some(offset) = output[search_from..].find(&branch_marker) {
        let branch_start = search_from + offset;
        let body = &output[branch_start + branch_marker.len()..];
        // A branch ends at the next branch, or at the end of its `when` for the last one.
        let body_end = ["\n            is ", "\n        }\n"]
            .iter()
            .filter_map(|terminator| body.find(terminator))
            .min()
            .ok_or_else(|| anyhow!("Unterminated converter branch for {class_name}.{variant_name}"))?;
        let branch_end = branch_start + branch_marker.len() + body_end;
        let branch = &output[branch_start..branch_end];
        if branch.contains(&old_access) {
            let renamed = branch.replace(&old_access, &new_access);
            output.replace_range(branch_start..branch_end, &renamed);
            renamed_branches += 1;
        }
        search_from = branch_start + branch_marker.len();
    }
    if renamed_branches != 2 {
        bail!(
            "Expected the allocationSize and write branches of {class_name}.{variant_name}, renamed {renamed_branches}"
        );
    }
    Ok(output)
}

/// UniFFI's Kotlin name for an error type or variant: an `Error` suffix becomes `Exception`.
fn exception_name(name: &str) -> String {
    name.strip_suffix("Error").map_or_else(|| name.to_string(), |stem| format!("{stem}Exception"))
}
