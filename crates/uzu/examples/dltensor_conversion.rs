/// Example demonstrating DLTensor to MetalArray conversion
///
/// Run with: cargo run --example dltensor_conversion
use metal::Device as MTLDevice;
use uzu::{Array, DLTensorExt, DataType};
use xgrammar::{DLDataType, DLDevice, DLDeviceType, DLTensor};

fn main() {
    println!("DLTensor to MetalArray Conversion Example\n");

    // Get Metal device
    let device =
        MTLDevice::system_default().expect("Failed to get Metal device");

    // Example 1: Create a simple DLTensor
    println!("=== Example 1: Simple Float32 Tensor ===");
    let mut data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut shape: Vec<i64> = vec![2, 3];

    let dltensor = DLTensor {
        data: data.as_mut_ptr() as *mut core::ffi::c_void,
        device: DLDevice {
            device_type: DLDeviceType::kDLMetal,
            device_id: 0,
        },
        ndim: 2,
        dtype: DLDataType {
            code: 2, // kDLFloat
            bits: 32,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: std::ptr::null_mut(), // Contiguous
        byte_offset: 0,
    };

    // Convert to MetalArray
    let metal_array = unsafe {
        dltensor
            .to_metal_array(&device)
            .expect("Failed to convert DLTensor to MetalArray")
    };

    println!("Created MetalArray:");
    println!("  Shape: {:?}", metal_array.shape());
    println!("  DataType: {:?}", metal_array.data_type());
    println!("  Size: {} bytes", metal_array.size_in_bytes());

    // Example 2: Using DataType conversion
    println!("\n=== Example 2: DataType to DLDataType Conversion ===");
    let uzu_types = vec![
        DataType::F32,
        DataType::F16,
        DataType::BF16,
        DataType::I32,
        DataType::U8,
    ];

    for dtype in uzu_types {
        let dl_dtype: DLDataType = dtype.into();
        println!(
            "{:?} -> DLDataType(code={}, bits={}, lanes={})",
            dtype, dl_dtype.code, dl_dtype.bits, dl_dtype.lanes
        );
    }

    // Example 3: Error handling
    println!("\n=== Example 3: Error Handling ===");

    // Try to convert a CPU tensor (should fail)
    let cpu_tensor = DLTensor {
        data: data.as_mut_ptr() as *mut core::ffi::c_void,
        device: DLDevice {
            device_type: DLDeviceType::kDLCPU,
            device_id: 0,
        },
        ndim: 1,
        dtype: DLDataType {
            code: 2,
            bits: 32,
            lanes: 1,
        },
        shape: shape.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    };

    match unsafe { cpu_tensor.to_metal_array(&device) } {
        Ok(_) => println!("Unexpected success for CPU tensor"),
        Err(e) => println!("Expected error for CPU tensor: {}", e),
    }

    // Try to convert a vectorized type (should fail)
    let vec_tensor = DLTensor {
        data: data.as_mut_ptr() as *mut core::ffi::c_void,
        device: DLDevice {
            device_type: DLDeviceType::kDLMetal,
            device_id: 0,
        },
        ndim: 1,
        dtype: DLDataType {
            code: 2,
            bits: 32,
            lanes: 4, // float4
        },
        shape: shape.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    };

    match unsafe { vec_tensor.to_metal_array(&device) } {
        Ok(_) => println!("Unexpected success for vectorized tensor"),
        Err(e) => println!("Expected error for vectorized type: {}", e),
    }

    println!("\n=== Example Complete ===");
}
