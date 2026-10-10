use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    os::unix::fs::MetadataExt,
    path::Path,
};

use super::{
    MXU_LAYOUT, ParameterLoaderError,
    safetensors_metadata::{Dtype, HashMetadata, TensorInfo, read_metadata},
};
use crate::{
    backends::common::{Backend, kernel::matmul::pack_mxu_w4_tiles},
    config::weight_matrix::{AnyWeightMatrixSpec, Layout},
};

fn eligible(
    key: &str,
    tensor: &TensorInfo,
    header: &HashMetadata,
) -> bool {
    let Some(prefix) = key.strip_suffix(".weights") else {
        return false;
    };
    let linear = key.contains(".weights.") || key.contains(".output_embedding.");
    if !linear || key.contains(".input_embedding.") || tensor.dtype != Dtype::U8 || tensor.shape.len() != 2 {
        return false;
    }
    let (n, k) = (tensor.shape[0], tensor.shape[1] as u64 * 2);
    if !matches!((n, k), (34816, 5120) | (5120, 17408) | (16480, 5120) | (5120, 6144) | (14336, 5120) | (248320, 5120))
        || tensor.data_offsets.1.checked_sub(tensor.data_offsets.0) != Some(n as usize * k as usize / 2)
    {
        return false;
    }
    let spec = header.metadata.as_ref().and_then(|m| m.get(&format!("{prefix}.spec")));
    matches!(spec.and_then(|s| serde_json::from_str::<AnyWeightMatrixSpec>(s).ok()),
        Some(AnyWeightMatrixSpec::IntSpec(spec)) if spec.bits == 4 && spec.group_size == 64
            && !spec.is_symmetric && spec.layout == Layout::OutputInput)
}

pub(super) fn open<B: Backend>(
    model_path: &Path,
    mut source: File,
) -> Result<File, ParameterLoaderError<B>> {
    let (source_offset, mut header) = read_metadata(&source)?;
    let packed: Vec<_> = header
        .tensors
        .iter()
        .filter(|(key, tensor)| eligible(key, tensor, &header))
        .map(|(key, tensor)| (key.clone(), tensor.clone()))
        .collect();
    if packed.is_empty() {
        return Ok(source);
    }

    let stat = source.metadata()?;
    let fingerprint =
        format!("{MXU_LAYOUT}:{}:{}:{}:{}:{}", stat.dev(), stat.ino(), stat.len(), stat.mtime(), stat.mtime_nsec());
    let metadata = header.metadata.get_or_insert_default();
    metadata.insert("uzu_m5_max_source".into(), fingerprint);
    for (key, _) in &packed {
        metadata.insert(format!("{key}.mxu_layout"), MXU_LAYOUT.into());
    }
    let data_size = stat
        .len()
        .checked_sub(source_offset as u64)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated model weights"))?;
    let cache_path = model_path.join("model_m5_max.safetensors");
    let cached = || -> Option<File> {
        let file = File::open(&cache_path).ok()?;
        let (offset, actual) = read_metadata(&file).ok()?;
        (actual == header && file.metadata().ok()?.len() == offset as u64 + data_size).then_some(file)
    };
    if let Some(file) = cached() {
        return Ok(file);
    }

    // Serialize builders, then publish only a complete file. Readers keep their open inode.
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(model_path.join("model_m5_max.safetensors.lock"))?;
    lock.lock()?;
    if let Some(file) = cached() {
        return Ok(file);
    }

    let mut json = serde_json::to_vec(&header)?;
    json.resize(json.len().next_multiple_of(8), b' ');
    let destination_offset = 8 + json.len() as u64;
    let mut temporary = tempfile::NamedTempFile::new_in(model_path)?;
    temporary.write_all(&(json.len() as u64).to_le_bytes())?;
    temporary.write_all(&json)?;
    source.seek(SeekFrom::Start(source_offset as u64))?;
    if io::copy(&mut source, &mut temporary)? != data_size {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "model weights changed during preprocessing").into());
    }
    for (_, tensor) in packed {
        let (begin, end) = tensor.data_offsets;
        let mut original = vec![0u32; (end - begin) / 4];
        source.seek(SeekFrom::Start(source_offset as u64 + begin as u64))?;
        source.read_exact(bytemuck::cast_slice_mut(&mut original))?;
        let mut reordered = vec![0u32; original.len()];
        pack_mxu_w4_tiles(&original, &mut reordered, tensor.shape[0], tensor.shape[1] * 2);
        temporary.seek(SeekFrom::Start(destination_offset + begin as u64))?;
        temporary.write_all(bytemuck::cast_slice(&reordered))?;
    }
    let after = source.metadata()?;
    if after.len() != stat.len() || after.mtime() != stat.mtime() || after.mtime_nsec() != stat.mtime_nsec() {
        return Err(io::Error::other("model weights changed during preprocessing").into());
    }
    temporary.as_file().sync_all()?;
    Ok(temporary.persist(&cache_path).map_err(|error| error.error)?)
}
