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
        let rpm = maximum.as_f32()?;
        if !rpm.is_finite() || rpm <= 0.0 {
            return Err(SmcError::InvalidData {
                key: maximum.key,
                reason: "invalid maximum fan RPM",
            });
        }
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
}
