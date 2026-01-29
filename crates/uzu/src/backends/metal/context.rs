use std::{cell::RefCell, collections::HashMap, env, rc::Rc};

use metal::{MTLBuffer, MTLCommandBuffer};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{
    Metal, MetalArray, error::MTLError, kernel,
    metal_extensions::LibraryPipelineExtensions,
};
use crate::{
    DataType, DeviceContext,
    array::array_size_in_bytes,
    backends::common::Context,
    backends::metal::{
        MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLDeviceExt,
        MTLFunctionConstantValues, MTLLibrary, MTLResourceExt,
        MTLResourceOptions,
    },
};

/// Apple GPU architecture generation.
/// Based on Apple GPU family naming convention (e.g., "applegpu_g13p").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceGeneration {
    Gen13, // M1 family
    Gen14, // M2 family
    Gen15, // M3 family
    Gen16, // M3 Pro/Max with enhanced features
    Gen17, // M4 family
    Gen18, // M5 family (NAX capable)
    Unknown(u8),
}

impl DeviceGeneration {
    pub fn generation_number(&self) -> u8 {
        match self {
            Self::Gen13 => 13,
            Self::Gen14 => 14,
            Self::Gen15 => 15,
            Self::Gen16 => 16,
            Self::Gen17 => 17,
            Self::Gen18 => 18,
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
    pub fn from_device(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let arch_string = device.name();

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

    fn parse_architecture_info(arch: &str) -> (DeviceGeneration, DeviceClass) {
        // Parse generation and class from device name (e.g. "Apple M3 Pro")
        let generation = if arch.contains("M5") {
            DeviceGeneration::Gen18
        } else if arch.contains("M4") {
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
    /// NAX requires M5 or later (generation >= 18) and macOS 26.2+.
    /// Since we can't check macOS version at compile time in Rust,
    /// we check generation only - the kernels will fail gracefully if unavailable.
    pub fn is_nax_available(&self) -> bool {
        self.generation.generation_number() >= 18
    }

    /// Returns true if this is a high-performance device (Pro/Max class).
    pub fn is_high_performance(&self) -> bool {
        self.device_class.is_high_performance()
    }
}

pub struct MTLContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub architecture: DeviceArchitecture,
    pub library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipeline_cache: RefCell<
        HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    >,
}

impl MTLContext {
    /// Returns true if NAX kernels are available on this device.
    pub fn is_nax_available(&self) -> bool {
        cfg!(feature = "metal-nax") && self.architecture.is_nax_available()
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

    /// TF32 toggle via UZU_TF32 environment variable.
    pub fn tf32_enabled(&self) -> bool {
        env::var("UZU_TF32")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    pub fn compute_pipeline_state(
        &self,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MTLError>
    {
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
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MTLError>
    {
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
}

impl Context for MTLContext {
    type Backend = Metal;

    fn new() -> Result<Rc<Self>, MTLError> {
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            <dyn metal::MTLDevice>::system_default().ok_or(
                MTLError::Generic("cannot open system default device".into()),
            )?;

        let command_queue = device
            .new_command_queue_with_max_command_buffer_count(1024)
            .ok_or(MTLError::Generic("cannot create command queue".into()))?;

        let library =
            device.new_library_with_data(kernel::MTLB).map_err(|nserror| {
                MTLError::Generic(format!(
                    "Failed to create Metal library: {}",
                    nserror
                ))
            })?;

        let architecture = DeviceArchitecture::from_device(&device);

        Ok(Rc::new(Self {
            device,
            command_queue,
            architecture,
            library,
            pipeline_cache: RefCell::new(HashMap::new()),
        }))
    }

    fn allocate_buffer(
        &self,
        size: u64,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MTLError> {
        self.device
            .new_buffer(size as usize, MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or(MTLError::Generic("cannot allocate buffer".into()))
    }

    fn allocate_command_buffer(
        &self
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, MTLError> {
        self.command_queue.command_buffer().ok_or(MTLError::Generic(
            "cannot to allocate command buffer".into(),
        ))
    }
}

impl DeviceContext for MTLContext {
    type DeviceArray = MetalArray;

    unsafe fn array_uninitialized(
        &self,
        shape: &[usize],
        data_type: DataType,
        label: String,
    ) -> MetalArray {
        unsafe {
            let buffer_size_bytes = array_size_in_bytes(shape, data_type);

            let buffer = self
                .allocate_buffer(buffer_size_bytes as u64)
                .expect("Failed to create buffer");
            buffer.set_label(Some(&label));
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
        label: String,
    ) -> MetalArray {
        unsafe { (**self).array_uninitialized(shape, data_type, label) }
    }
}
