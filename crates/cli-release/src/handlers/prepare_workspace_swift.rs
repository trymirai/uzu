use std::fs;

use crate::{
    handlers::{prepare_docs, prepare_platform},
    types::{Environment, Error, Language},
    utilities::{
        clone_dir_with_ignore_respect, extract_examples, extract_snippets, update_package_swift, update_readme,
    },
};

pub fn prepare_workspace_swift(
    environment: &Environment,
    version: String,
    checksum: String,
) -> Result<(), Error> {
    let workspace = environment.workspace();
    let workspace_swift_path = workspace.swift_path();
    if workspace_swift_path.exists() {
        fs::remove_dir_all(workspace_swift_path.clone()).map_err(|_| Error::UnableToPrepareWorkspaceSwift)?;
    }
    fs::create_dir_all(workspace_swift_path.clone()).map_err(|_| Error::UnableToPrepareWorkspaceSwift)?;

    let bindings_swift_path = environment.bindings().swift_path();
    if !bindings_swift_path.exists() {
        return Err(Error::PathNotFound {
            path: bindings_swift_path,
        });
    }

    clone_dir_with_ignore_respect(&bindings_swift_path, &workspace_swift_path)?;

    let package_swift_path = workspace_swift_path.join("Package.swift");
    update_package_swift(package_swift_path.clone(), version.clone(), checksum.clone())?;

    let examples_path = workspace_swift_path.join("Sources").join("Example");
    let snippets = extract_snippets(examples_path.clone())?;
    let examples = extract_examples(examples_path.clone(), workspace_swift_path.clone())?;

    let readme_path = workspace_swift_path.join("README.md");
    update_readme(readme_path.clone(), snippets.clone(), examples.clone(), version.clone())?;

    prepare_docs(environment, version.clone(), Language::Swift, snippets.clone(), examples.clone())?;

    prepare_platform(environment, Language::Swift, examples.clone())?;

    Ok(())
}
