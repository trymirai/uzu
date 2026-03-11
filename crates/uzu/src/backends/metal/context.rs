use std::{cell::RefCell, collections::HashMap, env, rc::Rc};

use metal::{
    MTLBuffer, MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager, MTLCommandQueue, MTLCommandQueueExt,
    MTLComputePipelineState, MTLDevice, MTLDeviceExt, MTLEvent, MTLFunctionConstantValues, MTLLibrary,
    MTLResourceOptions,
};
use objc2::{msg_send, rc::Retained, runtime::ProtocolObject};

use super::{Metal, error::MetalError, kernel, metal_extensions::LibraryPipelineExtensions};
use crate::backends::{
    common::{Context, DeviceClass as CommonDeviceClass},
    metal::command_buffer::MetalCommandBufferInitial,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceGeneration {
    Gen13,
    Gen14,
    Gen15,
    Gen16,
    Gen17,
    Gen18,
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

    fn from_generation_number(generation_number: u8) -> Self {
        match generation_number {
            13 => Self::Gen13,
            14 => Self::Gen14,
            15 => Self::Gen15,
            16 => Self::Gen16,
            17 => Self::Gen17,
            18 => Self::Gen18,
            _ => Self::Unknown(generation_number),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTier {
    Phone,
    Base,
    Max,
    Ultra,
    Unknown(char),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuArchitecture {
    pub generation: DeviceGeneration,
    pub tier: GpuTier,
}

impl GpuArchitecture {
    fn parse(architecture_name: &str) -> Option<Self> {
        let suffix_start = architecture_name.find("applegpu_g")?;
        let after_prefix = &architecture_name[suffix_start + "applegpu_g".len()..];

        let digit_end = after_prefix
            .char_indices()
            .find(|(_, character)| !character.is_ascii_digit())
            .map_or(after_prefix.len(), |(index, _)| index);

        if digit_end == 0 {
            return None;
        }

        let generation_number: u8 = after_prefix[..digit_end].parse().ok()?;
        let tier_character = after_prefix[digit_end..].chars().next()?;

        let tier = match tier_character {
            'p' => GpuTier::Phone,
            'g' => GpuTier::Base,
            's' => GpuTier::Max,
            'd' => GpuTier::Ultra,
            other => GpuTier::Unknown(other),
        };

        Some(Self {
            generation: DeviceGeneration::from_generation_number(generation_number),
            tier,
        })
    }

    fn unknown() -> Self {
        Self {
            generation: DeviceGeneration::Unknown(0),
            tier: GpuTier::Unknown('?'),
        }
    }
}

pub type DeviceClass = GpuTier;

impl GpuTier {
    pub fn is_high_performance(&self) -> bool {
        matches!(self, Self::Max | Self::Ultra)
    }

    fn common_device_class(self) -> CommonDeviceClass {
        match self {
            Self::Phone => CommonDeviceClass::IPhone,
            Self::Base | Self::Unknown(_) => CommonDeviceClass::Base,
            Self::Max => CommonDeviceClass::Max,
            Self::Ultra => CommonDeviceClass::Ultra,
        }
    }
}

fn gpu_architecture_name(device: &ProtocolObject<dyn MTLDevice>) -> Option<String> {
    unsafe {
        let architecture: *const objc2::runtime::AnyObject = msg_send![device, architecture];
        if architecture.is_null() {
            return None;
        }
        let name: *const objc2::runtime::AnyObject = msg_send![&*architecture, name];
        if name.is_null() {
            return None;
        }
        let utf8: *const std::ffi::c_char = msg_send![&*name, UTF8String];
        if utf8.is_null() {
            return None;
        }
        Some(std::ffi::CStr::from_ptr(utf8).to_string_lossy().into_owned())
    }
}

pub struct DeviceArchitecture {
    pub generation: DeviceGeneration,
    pub tier: GpuTier,
    #[allow(dead_code)]
    architecture_name: String,
}

impl DeviceArchitecture {
    pub fn from_device(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let architecture_name = gpu_architecture_name(device).unwrap_or_default();
        let gpu_architecture = GpuArchitecture::parse(&architecture_name).unwrap_or_else(GpuArchitecture::unknown);

        Self {
            generation: gpu_architecture.generation,
            tier: gpu_architecture.tier,
            architecture_name,
        }
    }

    pub fn is_high_performance(&self) -> bool {
        self.tier.is_high_performance()
    }
}

pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    architecture: DeviceArchitecture,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipeline_cache: RefCell<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
}

impl MetalContext {
    pub fn is_high_performance(&self) -> bool {
        self.architecture.is_high_performance()
    }

    pub fn device_generation(&self) -> DeviceGeneration {
        self.architecture.generation
    }

    pub fn device_class(&self) -> DeviceClass {
        self.architecture.tier
    }

    pub fn tf32_enabled(&self) -> bool {
        env::var("UZU_TF32").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false)
    }

    pub fn compute_pipeline_state(
        &self,
        cache_key: &str,
        function_name: &str,
        constants: Option<&MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
        if let Some(pipeline) = self.pipeline_cache.borrow().get(cache_key) {
            return Ok(pipeline.clone());
        }

        let pipeline = self.library.compute_pipeline_state(function_name, constants)?;

        self.pipeline_cache.borrow_mut().insert(cache_key.to_string(), pipeline.clone());

        Ok(pipeline)
    }
}

impl Context for MetalContext {
    type Backend = Metal;

    fn new() -> Result<Rc<Self>, MetalError> {
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            <dyn metal::MTLDevice>::system_default().ok_or(MetalError::CannotOpenDevice)?;

        let command_queue =
            device.new_command_queue_with_max_command_buffer_count(1024).ok_or(MetalError::CannotCreateCommandQueue)?;

        let library = device
            .new_library_with_data(kernel::MTLB)
            .map_err(|nserror| MetalError::CannotCreateLibrary(nserror.to_string()))?;

        let architecture = DeviceArchitecture::from_device(&device);

        Ok(Rc::new(Self {
            device,
            command_queue,
            architecture,
            library,
            pipeline_cache: RefCell::new(HashMap::new()),
        }))
    }

    fn device_class(&self) -> CommonDeviceClass {
        self.architecture.tier.common_device_class()
    }

    fn debug_active(&self) -> bool {
        let upper = std::env::var("METAL_DEVICE_WRAPPER_TYPE").unwrap_or_default().to_ascii_uppercase();
        matches!(upper.as_str(), "1" | "YES" | "TRUE")
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        self.device.new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED).ok_or(MetalError::CannotCreateBuffer)
    }

    fn create_command_buffer(&self) -> Result<MetalCommandBufferInitial, MetalError> {
        Ok(MetalCommandBufferInitial::new(
            self.command_queue.command_buffer().ok_or(MetalError::CannotCreateCommandBuffer)?,
        ))
    }

    fn create_event(&self) -> Result<Retained<ProtocolObject<dyn MTLEvent>>, MetalError> {
        self.device.new_event().ok_or(MetalError::CannotCreateEvent)
    }

    fn enable_capture() {
        unsafe {
            std::env::set_var("METAL_CAPTURE_ENABLED", "1");
        }
    }

    fn start_capture(
        &self,
        trace_path: &std::path::Path,
    ) -> Result<(), <Self::Backend as crate::backends::common::Backend>::Error> {
        let capture_manager = MTLCaptureManager::shared_capture_manager();
        let capture_descriptor = MTLCaptureDescriptor::new();
        capture_descriptor.set_destination(MTLCaptureDestination::GPUTraceDocument);
        capture_descriptor.set_output_path(Some(trace_path));

        self.command_queue.set_label(Some("uzu_command_queue"));
        capture_descriptor.set_capture_object(Some(self.command_queue.as_ref()));

        capture_manager
            .start_capture_with_descriptor_error(&capture_descriptor)
            .map_err(|nserror| MetalError::CannotStartGpuCapture(nserror.to_string()))?;

        Ok(())
    }

    fn stop_capture(&self) -> Result<(), <Self::Backend as crate::backends::common::Backend>::Error> {
        MTLCaptureManager::shared_capture_manager().stop_capture();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{DeviceGeneration, GpuArchitecture, GpuTier};
    use crate::backends::common::DeviceClass as CommonDeviceClass;

    #[test]
    fn parses_phone_architecture() {
        let architecture = GpuArchitecture::parse("applegpu_g13p").unwrap();

        assert_eq!(architecture.generation, DeviceGeneration::Gen13);
        assert_eq!(architecture.tier, GpuTier::Phone);
        assert_eq!(architecture.tier.common_device_class(), CommonDeviceClass::IPhone);
    }

    #[test]
    fn parses_base_mac_architecture() {
        let architecture = GpuArchitecture::parse("applegpu_g14g").unwrap();

        assert_eq!(architecture.generation, DeviceGeneration::Gen14);
        assert_eq!(architecture.tier, GpuTier::Base);
        assert_eq!(architecture.tier.common_device_class(), CommonDeviceClass::Base);
    }

    #[test]
    fn parses_max_mac_architecture() {
        let architecture = GpuArchitecture::parse("applegpu_g17s").unwrap();

        assert_eq!(architecture.generation, DeviceGeneration::Gen17);
        assert_eq!(architecture.tier, GpuTier::Max);
        assert_eq!(architecture.tier.common_device_class(), CommonDeviceClass::Max);
    }

    #[test]
    fn parses_ultra_mac_architecture() {
        let architecture = GpuArchitecture::parse("applegpu_g18d").unwrap();

        assert_eq!(architecture.generation, DeviceGeneration::Gen18);
        assert_eq!(architecture.tier, GpuTier::Ultra);
        assert_eq!(architecture.tier.common_device_class(), CommonDeviceClass::Ultra);
    }

    #[test]
    fn parses_m3_base_architecture() {
        let architecture = GpuArchitecture::parse("applegpu_g15g").unwrap();

        assert_eq!(architecture.generation, DeviceGeneration::Gen15);
        assert_eq!(architecture.tier, GpuTier::Base);
    }

    #[test]
    fn parses_m3_pro_max_architecture() {
        let architecture = GpuArchitecture::parse("applegpu_g16s").unwrap();

        assert_eq!(architecture.generation, DeviceGeneration::Gen16);
        assert_eq!(architecture.tier, GpuTier::Max);
    }

    #[test]
    fn returns_none_for_malformed_architecture() {
        assert!(GpuArchitecture::parse("").is_none());
        assert!(GpuArchitecture::parse("not_a_gpu").is_none());
        assert!(GpuArchitecture::parse("applegpu_g").is_none());
        assert!(GpuArchitecture::parse("applegpu_gp").is_none());
    }

    #[test]
    fn returns_unknown_for_unrecognized_generation() {
        let architecture = GpuArchitecture::parse("applegpu_g99g").unwrap();

        assert_eq!(architecture.generation, DeviceGeneration::Unknown(99));
        assert_eq!(architecture.tier, GpuTier::Base);
    }

    #[test]
    fn returns_unknown_tier_for_unrecognized_suffix() {
        let architecture = GpuArchitecture::parse("applegpu_g18z").unwrap();

        assert_eq!(architecture.generation, DeviceGeneration::Gen18);
        assert_eq!(architecture.tier, GpuTier::Unknown('z'));
    }

    #[test]
    fn high_performance_only_for_max_and_ultra() {
        assert!(!GpuTier::Phone.is_high_performance());
        assert!(!GpuTier::Base.is_high_performance());
        assert!(GpuTier::Max.is_high_performance());
        assert!(GpuTier::Ultra.is_high_performance());
        assert!(!GpuTier::Unknown('x').is_high_performance());
    }
}
