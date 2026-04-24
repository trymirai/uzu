use std::{fs, path::PathBuf};

use crate::types::Error;

pub fn update_package_swift(
    package_swift_path: PathBuf,
    version: String,
    checksum: String,
) -> Result<(), Error> {
    let original_content = fs::read_to_string(&package_swift_path).map_err(|_| Error::UnableToUpdatePackageSwift)?;

    let framework_url = format!("https://artifacts.trymirai.com/uzu-swift/releases/{}.zip", version);

    let dependency_snippet = format!("url: \"{}\",\n            checksum: \"{}\"", framework_url, checksum);

    let updated_content = original_content.replace("path: \"uzu.xcframework\"", dependency_snippet.as_str());

    fs::write(&package_swift_path, updated_content).map_err(|_| Error::UnableToUpdatePackageSwift)?;
    Ok(())
}
