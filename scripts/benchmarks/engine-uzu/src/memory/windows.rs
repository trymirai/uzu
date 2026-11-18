use std::cell::RefCell;

use windows::{
    Win32::{
        Graphics::Dxgi::{
            CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
            DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3, IDXGIFactory1,
        },
        System::{
            ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX},
            Threading::GetCurrentProcess,
        },
    },
    core::Interface,
};

/// This process's memory as Windows accounts it: private commit for the footprint, the working set for the resident
/// size, and the local video memory it uses on its GPU for graphics.
pub struct MemoryCounters {
    pub phys_footprint: u64,
    pub resident_size: u64,
    pub graphics_total: u64,
}

thread_local! {
    // Opened once per thread: the adapter samples memory after every token.
    static ADAPTERS: RefCell<Option<Vec<IDXGIAdapter3>>> = const { RefCell::new(None) };
}

impl MemoryCounters {
    pub fn release_thread_cache() {
        ADAPTERS.with_borrow_mut(|adapters| *adapters = None);
    }

    pub fn collect() -> anyhow::Result<Self> {
        let mut process = PROCESS_MEMORY_COUNTERS_EX {
            cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
            ..Default::default()
        };
        // SAFETY: the counters are writable and `cb` declares their extended layout.
        unsafe {
            GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut process).cast::<PROCESS_MEMORY_COUNTERS>(), process.cb)
        }?;
        Ok(Self {
            phys_footprint: process.PrivateUsage as u64,
            resident_size: process.WorkingSetSize as u64,
            graphics_total: local_video_memory()?,
        })
    }
}

fn local_video_memory() -> anyhow::Result<u64> {
    ADAPTERS.with_borrow_mut(|adapters| {
        let adapters = match adapters {
            Some(adapters) => adapters,
            None => adapters.insert(hardware_adapters()?),
        };
        // uzu runs on one GPU, and DXGI can list that GPU under several LUIDs (a Remote Desktop session adds one), each
        // reporting this process's usage: summing them would count it more than once.
        let mut used = 0;
        for adapter in adapters.iter() {
            let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            // SAFETY: node 0 exists on every adapter and `info` is writable.
            unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }?;
            used = used.max(info.CurrentUsage);
        }
        Ok(used)
    })
}

fn hardware_adapters() -> anyhow::Result<Vec<IDXGIAdapter3>> {
    // SAFETY: plain DXGI factory and adapter enumeration.
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }?;
    let mut adapters = Vec::new();
    for index in 0.. {
        let adapter = match unsafe { factory.EnumAdapters1(index) } {
            Ok(adapter) => adapter,
            Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(error) => return Err(error.into()),
        };
        if unsafe { adapter.GetDesc1() }?.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            adapters.push(adapter.cast()?);
        }
    }
    Ok(adapters)
}
