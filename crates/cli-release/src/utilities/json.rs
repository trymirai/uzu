use std::{fs, path::PathBuf};

use serde_json::Value;

use crate::types::Error;

pub fn update_json_field(
    path: PathBuf,
    key: String,
    value: String,
) -> Result<(), Error> {
    let json = fs::read_to_string(path.clone()).map_err(|_| Error::UnableToUpdateJSON)?;
    let mut json: Value = serde_json::from_str(&json).map_err(|_| Error::UnableToUpdateJSON)?;
    json[key] = Value::String(value);
    fs::write(path.clone(), serde_json::to_string(&json).map_err(|_| Error::UnableToUpdateJSON)?)
        .map_err(|_| Error::UnableToUpdateJSON)?;

    Ok(())
}
