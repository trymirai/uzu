use obfstr::obfstr;

use super::{Smc, SmcError, SmcKeyData};

pub struct FanState {
    pub mode: SmcKeyData,
    pub target: SmcKeyData,
    pub maximum: f32,
}

impl FanState {
    pub fn read(
        smc: &Smc,
        index: u8,
    ) -> Result<Self, SmcError> {
        let key = |suffix: &str| format!("{}{}{}", obfstr!("F"), index, suffix);
        let mode = match smc.read_key(&key(obfstr!("md"))) {
            Err(SmcError::MissingKey(_)) => smc.read_key(&key(obfstr!("Md")))?,
            other => other?,
        };
        if !matches!(mode.as_u8()?, 0 | 1 | 3) {
            return Err(SmcError::InvalidData {
                key: mode.key,
                reason: "unsupported fan mode",
            });
        }
        let maximum = smc.read_key(&key(obfstr!("Mx")))?;
        let rpm = Self::maximum(&maximum)?;
        let target = smc.read_key(&key(obfstr!("Tg")))?;
        let target_rpm = target.as_f32()?;
        if !target_rpm.is_finite() || target_rpm < 0.0 || target_rpm > rpm {
            return Err(SmcError::InvalidData {
                key: target.key,
                reason: "invalid fan target",
            });
        }
        Ok(Self {
            mode,
            target,
            maximum: rpm,
        })
    }

    fn maximum(value: &SmcKeyData) -> Result<f32, SmcError> {
        let rpm = value.as_f32()?;
        if !rpm.is_finite() || rpm <= 0.0 {
            return Err(SmcError::InvalidData {
                key: value.key,
                reason: "invalid maximum fan RPM",
            });
        }
        Ok(rpm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys::smc::{SmcKeyInfo, fourcc};

    #[test]
    fn maximum_rejects_invalid_hardware_limits() {
        for rpm in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            let mut value = SmcKeyData {
                key_info: SmcKeyInfo {
                    data_type: fourcc("flt ").unwrap(),
                    data_size: 4,
                    ..Default::default()
                },
                ..Default::default()
            };
            value.bytes[..4].copy_from_slice(&rpm.to_le_bytes());
            assert!(matches!(FanState::maximum(&value), Err(SmcError::InvalidData { .. })));
        }
    }
}
