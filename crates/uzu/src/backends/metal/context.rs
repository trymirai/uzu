#![allow(dead_code)]
include!(concat!(env!("OUT_DIR"), "/metal_lib.rs"));

use std::{cell::RefCell, collections::HashMap, rc::Rc, env};

use metal::{
    CommandQueue as MTLCommandQueue,
    ComputePipelineState as MTLComputePipelineState, Device as MTLDevice,
    FunctionConstantValues, Library as MTLLibrary,
};

use super::{
    MetalArray, error::MTLError, metal_extensions::LibraryPipelineExtensions,
};
use crate::{DataType, DeviceContext, array::array_size_in_bytes};

/// Apple GPU architecture generation.
/// Based on Apple GPU family naming convention (e.g., "applegpu_g13p").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceGeneration {
    Gen13, // M1 family
    Gen14, // M2 family
    Gen15, // M3 family
    Gen16, // M3 Pro/Max with enhanced features
    Gen17, // M4 family (NAX capable)
    Unknown(u8),
}

impl DeviceGeneration {
    fn from_generation_number(r#gen: u8) -> Self {
        match r#gen {
            13 => Self::Gen13,
            14 => Self::Gen14,
            15 => Self::Gen15,
            16 => Self::Gen16,
            17 => Self::Gen17,
            n => Self::Unknown(n),
        }
    }

    pub fn generation_number(&self) -> u8 {
        match self {
            Self::Gen13 => 13,
            Self::Gen14 => 14,
            Self::Gen15 => 15,
            Self::Gen16 => 16,
            Self::Gen17 => 17,
            Self::Unknown(n) => *n,
        }
    }
}

/// Device performance class based on the last character of architecture name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceClass {
    Phone,      // 'p' - iPhone/iPad integrated
    Integrated, // 'g' - Mac integrated GPU
    Desktop,    // 'd' - Mac Pro/Max discrete-class GPU
    Unknown(char),
}

impl DeviceClass {
    fn from_arch_suffix(c: char) -> Self {
        match c {
            'p' => Self::Phone,
            'g' => Self::Integrated,
            'd' => Self::Desktop,
            other => Self::Unknown(other),
        }
    }

    pub fn is_high_performance(&self) -> bool {
        matches!(self, Self::Desktop)
    }
}

/// Complete device architecture information.
#[derive(Debug, Clone)]
pub struct DeviceArchitecture {
    pub generation: DeviceGeneration,
    pub device_class: DeviceClass,
    pub arch_string: String,
}

impl DeviceArchitecture {
    pub fn from_device(device: &MTLDevice) -> Self {
        let arch_string = device.name().to_string();

        // Extract generation from architecture name
        // Format is typically like "Apple M3 Pro" or the GPU name "applegpu_g15d"
        let (generation, device_class) =
            Self::parse_architecture_info(&arch_string);

        Self {
            generation,
            device_class,
            arch_string,
        }
    }

    fn parse_architecture_info(
        arch: &str,
    ) -> (DeviceGeneration, DeviceClass) {
        // Try to extract from GPU architecture string if available
        // MLX uses device_->architecture()->name() which gives "applegpu_gXXY"
        // where XX is generation and Y is class (p/g/d)
        // The metal crate's device.name() gives the marketing name like "Apple M3 Pro"

        // For now, we'll use a heuristic based on the device name
        let generation = if arch.contains("M4") {
            DeviceGeneration::Gen17
        } else if arch.contains("M3") {
            if arch.contains("Max") || arch.contains("Ultra") {
                DeviceGeneration::Gen16
            } else {
                DeviceGeneration::Gen15
            }
        } else if arch.contains("M2") {
            DeviceGeneration::Gen14
        } else if arch.contains("M1") {
            DeviceGeneration::Gen13
        } else {
            DeviceGeneration::Unknown(0)
        };

        let device_class = if arch.contains("Max")
            || arch.contains("Ultra")
            || arch.contains("Pro")
        {
            DeviceClass::Desktop
        } else if arch.contains("iPhone") || arch.contains("iPad") {
            DeviceClass::Phone
        } else {
            DeviceClass::Integrated
        };

        (generation, device_class)
    }

    /// Returns true if NAX (New Accelerator eXtensions) is available.
    /// NAX requires M4 or later (generation >= 17) and macOS 26.2+.
    /// Since we can't check macOS version at compile time in Rust,
    /// we check generation only - the kernels will fail gracefully if unavailable.
    pub fn is_nax_available(&self) -> bool {
        self.generation.generation_number() >= 17
    }

    /// Returns true if this is a high-performance device (Pro/Max class).
    pub fn is_high_performance(&self) -> bool {
        self.device_class.is_high_performance()
    }
}

pub struct MTLContext {
    pub device: MTLDevice,
    pub command_queue: MTLCommandQueue,
    pub architecture: DeviceArchitecture,
    library: MTLLibrary,
    pipeline_cache: RefCell<HashMap<String, MTLComputePipelineState>>,
}

impl MTLContext {
    pub fn new(
        device: MTLDevice,
        command_queue: MTLCommandQueue,
    ) -> Result<Self, MTLError> {
        let library = match device.new_library_with_data(METAL_LIBRARY_DATA) {
            Ok(lib) => lib,
            Err(e) => {
                return Err(MTLError::Generic(format!(
                    "Failed to create Metal library: {}",
                    e
                )));
            },
        };

        let architecture = DeviceArchitecture::from_device(&device);

        Ok(Self {
            device,
            command_queue,
            architecture,
            library,
            pipeline_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Returns true if NAX kernels are available on this device.
    pub fn is_nax_available(&self) -> bool {
        self.architecture.is_nax_available()
    }

    /// Returns true if this is a high-performance device (Pro/Max class).
    pub fn is_high_performance(&self) -> bool {
        self.architecture.is_high_performance()
    }

    /// Returns the device generation for tile size selection.
    pub fn device_generation(&self) -> DeviceGeneration {
        self.architecture.generation
    }

    /// Returns the device class for performance tuning.
    pub fn device_class(&self) -> DeviceClass {
        self.architecture.device_class
    }

    /// TF32 toggle similar to MLX env::enable_tf32().
    pub fn tf32_enabled(&self) -> bool {
        env::var("UZU_TF32")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    pub fn compute_pipeline_state(
        &self,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<MTLComputePipelineState, MTLError> {
        // Only cache pipelines without constants
        if constants.is_some() {
            return self
                .library
                .compute_pipeline_state(function_name, constants);
        }
        self.compute_pipeline_state_cached(function_name, function_name, None)
    }

    pub fn compute_pipeline_state_cached(
        &self,
        cache_key: &str,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<MTLComputePipelineState, MTLError> {
        if let Some(pipeline) = self.pipeline_cache.borrow().get(cache_key) {
            return Ok(pipeline.clone());
        }

        let pipeline =
            self.library.compute_pipeline_state(function_name, constants)?;

        self.pipeline_cache
            .borrow_mut()
            .insert(cache_key.to_string(), pipeline.clone());

        Ok(pipeline)
    }

    pub fn compute_pipeline_state_with_reflection(
        &self,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<(MTLComputePipelineState, Vec<String>), MTLError> {
        // Only cache pipelines without constants
        if constants.is_some() {
            return self.library.compute_pipeline_state_with_reflection(
                function_name,
                constants,
            );
        }
        self.compute_pipeline_state_with_reflection_cached(
            function_name,
            function_name,
            None,
        )
    }

    pub fn compute_pipeline_state_with_reflection_cached(
        &self,
        cache_key: &str,
        function_name: &str,
        constants: Option<&FunctionConstantValues>,
    ) -> Result<(MTLComputePipelineState, Vec<String>), MTLError> {
        if let Some(pipeline) = self.pipeline_cache.borrow().get(cache_key) {
            return Ok((pipeline.clone(), Vec::new()));
        }

        let (pipeline, arg_names) = self
            .library
            .compute_pipeline_state_with_reflection(function_name, constants)?;

        self.pipeline_cache
            .borrow_mut()
            .insert(cache_key.to_string(), pipeline.clone());

        Ok((pipeline, arg_names))
    }
}

impl DeviceContext for MTLContext {
    type DeviceArray = MetalArray;

    unsafe fn array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
    ) -> MetalArray {
        unsafe {
            let buffer_size_bytes = array_size_in_bytes(shape, data_type);

            let buffer = self.device.new_buffer(
                buffer_size_bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            MetalArray::new(buffer, shape, data_type)
        }
    }
}

impl DeviceContext for Rc<MTLContext> {
    type DeviceArray = MetalArray;

    unsafe fn array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
    ) -> MetalArray {
        unsafe { (**self).array_uninitialized(shape, data_type) }
    }
}
