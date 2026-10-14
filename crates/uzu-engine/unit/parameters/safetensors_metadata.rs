use std::fs::File;

use uzu_engine_macros::uzu_test;

#[cfg(backend = "cpu")]
use crate::{
    backends::{
        common::{Backend, Context},
        cpu::Cpu,
    },
    data_type::DataType,
    parameters::ParameterLoader,
};
use crate::{
    parameters::safetensors_metadata::{read_metadata, read_metadata_from_bytes},
    tests::path::get_test_weights_path,
};

#[uzu_test]
#[ignore] // downloads the test model from the registry
fn test_metadata_loading() {
    let path = get_test_weights_path();
    let file = File::open(&path).expect("weights not found");
    let (_offset, metadata) = read_metadata(&file).expect("read metadata");
    assert!(!metadata.tensors.is_empty());
}

#[cfg(backend = "cpu")]
fn small_weights(end: usize) -> Vec<u8> {
    let header = serde_json::to_vec(&serde_json::json!({
        "weight": {"dtype":"F32", "shape":[2], "data_offsets":[0,end]}
    }))
    .unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&1.25f32.to_le_bytes());
    bytes.extend_from_slice(&(-2.5f32).to_le_bytes());
    bytes
}

#[uzu_test]
fn bytes_header_rejects_truncation_and_excessive_size() {
    for bytes in [&[][..], &[0; 7][..], &u64::MAX.to_le_bytes()[..], &[10, 0, 0, 0, 0, 0, 0, 0][..]] {
        assert!(read_metadata_from_bytes(bytes).is_err());
    }
}

#[uzu_test]
fn ranged_source_rejects_a_truncated_header_before_reading_its_body() {
    struct TruncatedHeader;
    impl crate::parameters::ParameterReadSource for TruncatedHeader {
        fn len(&self) -> u64 {
            8
        }
        fn read_exact_at(
            &self,
            destination: &mut [u8],
            offset: u64,
        ) -> std::io::Result<()> {
            assert_eq!(offset, 0, "truncated header body must not be allocated/read");
            destination.copy_from_slice(&(99_000_000u64).to_le_bytes());
            Ok(())
        }
    }
    assert!(crate::parameters::safetensors_metadata::read_metadata_from_source(&TruncatedHeader).is_err());
}

#[cfg(backend = "cpu")]
#[uzu_test]
fn bytes_parameter_loader_reads_tensor_without_a_file() {
    let bytes = small_weights(8);
    let context = <Cpu as Backend>::Context::new().unwrap();
    let loader = ParameterLoader::<Cpu>::from_bytes(&bytes, &context).unwrap();
    let tree = loader.tree();
    let tensor = tree.leaf("weight").unwrap().validate(&[2], DataType::F32).unwrap();
    assert_eq!(&*tensor.read_slice::<f32>().unwrap(), &[1.25, -2.5]);
    tree.assert_all_tensors_validated().unwrap();
}

#[cfg(backend = "cpu")]
#[uzu_test]
fn bytes_parameter_loader_rejects_out_of_bounds_tensors() {
    let bytes = small_weights(16);
    let context = <Cpu as Backend>::Context::new().unwrap();
    assert!(ParameterLoader::<Cpu>::from_bytes(&bytes, &context).is_err());
}

#[cfg(backend = "cpu")]
#[uzu_test]
fn ranged_source_reads_beyond_four_gib_without_large_allocation() {
    use crate::parameters::ParameterReadSource;
    struct SparseSource {
        header: Vec<u8>,
        tensor_offset: u64,
    }
    impl ParameterReadSource for SparseSource {
        fn len(&self) -> u64 {
            self.tensor_offset + 8
        }
        fn read_exact_at(
            &self,
            destination: &mut [u8],
            offset: u64,
        ) -> std::io::Result<()> {
            let source = if offset == self.tensor_offset {
                [1.25f32.to_le_bytes(), (-2.5f32).to_le_bytes()].concat()
            } else {
                self.header.get(usize::try_from(offset).unwrap_or(usize::MAX)..).unwrap_or_default().to_vec()
            };
            let bytes = source
                .get(..destination.len())
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "sparse range missing"))?;
            destination.copy_from_slice(bytes);
            Ok(())
        }
    }
    let local_offset = (1u64 << 32) + 123;
    let header = serde_json::to_vec(&serde_json::json!({
        "weight":{"dtype":"F32","shape":[2],"data_offsets":[local_offset,local_offset+8]}
    }))
    .unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    let source = SparseSource {
        tensor_offset: bytes.len() as u64 + local_offset,
        header: bytes,
    };
    let context = <Cpu as Backend>::Context::new().unwrap();
    let loader = ParameterLoader::<Cpu>::from_source(&source, &context, None).unwrap();
    let tree = loader.tree();
    assert_eq!(
        &*tree.leaf("weight").unwrap().validate(&[2], DataType::F32).unwrap().read_slice::<f32>().unwrap(),
        &[1.25, -2.5]
    );
    tree.assert_all_tensors_validated().unwrap();
}
