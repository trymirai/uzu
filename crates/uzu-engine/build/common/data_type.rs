use anyhow::bail;
use quote::format_ident;
use syn::Ident;

use super::Error;

/// The `DataType` of a Slang or Metal source type.
pub fn data_type(gpu_type: &str) -> Result<Ident, Error> {
    Ok(format_ident!(
        "{}",
        match gpu_type {
            "float" => "F32",
            "half" => "F16",
            "bf16" | "bfloat" => "BF16",
            "uint" => "U32",
            "int" => "I32",
            "uint8_t" => "U8",
            "int8_t" => "I8",
            other => bail!("no DataType for GPU type '{other}'"),
        }
    ))
}
