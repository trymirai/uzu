use metal::{MTLDevice, MTLDeviceExt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceClass {
    Ultra,
    Max,
    Pro,
    Base,
    IPhone,
    Unknown,
}

impl DeviceClass {
    pub fn detect() -> Self {
        let device = match <dyn MTLDevice>::system_default() {
            Some(dev) => dev,
            None => return DeviceClass::Unknown,
        };

        let name = device.name().to_lowercase();

        if name.contains("ultra") {
            DeviceClass::Ultra
        } else if name.contains("max") {
            DeviceClass::Max
        } else if name.contains("pro") {
            DeviceClass::Pro
        } else if name.contains("iphone") || name.contains("a1") {
            DeviceClass::IPhone
        } else {
            DeviceClass::Base
        }
    }

    pub fn is_high_end(&self) -> bool {
        matches!(self, DeviceClass::Ultra | DeviceClass::Max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_detection() {
        let device = DeviceClass::detect();
        println!("Detected device: {:?}", device);
        println!("Is high-end: {}", device.is_high_end());
        assert_ne!(device, DeviceClass::Unknown);
    }

    #[test]
    fn test_is_high_end() {
        assert!(DeviceClass::Ultra.is_high_end());
        assert!(DeviceClass::Max.is_high_end());
        assert!(!DeviceClass::Pro.is_high_end());
        assert!(!DeviceClass::Base.is_high_end());
        assert!(!DeviceClass::IPhone.is_high_end());
        assert!(!DeviceClass::Unknown.is_high_end());
    }
}
