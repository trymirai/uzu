use obfstr::obfstr;

pub(crate) trait RailKind: 'static {
    const RAIL: Rail;
    const TYPE_BIT: u128;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rail {
    Cpu,
    Gpu,
    Ane,
    Ram,
}

impl Rail {
    pub(crate) fn classify(name: &str) -> Option<Rail> {
        if name == obfstr!("GPU Energy") {
            Some(Rail::Gpu)
        } else if name.ends_with(obfstr!("CPU Energy")) {
            Some(Rail::Cpu)
        } else if name.starts_with(obfstr!("ANE")) {
            Some(Rail::Ane)
        } else if name.starts_with(obfstr!("DRAM"))
            || name.starts_with(obfstr!("DCS"))
            || name.starts_with(obfstr!("AMCC"))
        {
            Some(Rail::Ram)
        } else {
            None
        }
    }
}

pub struct Cpu;
pub struct Gpu;
pub struct Ane;
pub struct Ram;

impl RailKind for Cpu {
    const RAIL: Rail = Rail::Cpu;
    const TYPE_BIT: u128 = 1 << 0;
}

impl RailKind for Gpu {
    const RAIL: Rail = Rail::Gpu;
    const TYPE_BIT: u128 = 1 << 1;
}

impl RailKind for Ane {
    const RAIL: Rail = Rail::Ane;
    const TYPE_BIT: u128 = 1 << 2;
}

impl RailKind for Ram {
    const RAIL: Rail = Rail::Ram;
    const TYPE_BIT: u128 = 1 << 3;
}
