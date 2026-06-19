use core::ffi::{CStr, c_char};

use objc2_core_foundation::CFRetained;
use objc2_io_kit::{
    IOIteratorNext, IOObjectRelease, IORegistryEntryGetName, IOServiceGetMatchingServices, IOServiceMatching,
    io_iterator_t, io_object_t, io_registry_entry_t,
};

pub struct IoServiceIterator {
    iterator: io_iterator_t,
    current_entry: io_object_t,
}

impl IoServiceIterator {
    pub fn new(service_name: &str) -> Option<Self> {
        let name = std::ffi::CString::new(service_name).ok()?;

        let matching = unsafe { IOServiceMatching(name.as_ptr()) }?;
        let matching = unsafe { CFRetained::from_raw(CFRetained::into_raw(matching).cast()) };
        let mut iterator: io_iterator_t = 0;
        let result = unsafe { IOServiceGetMatchingServices(0, Some(matching), &mut iterator) };
        if result != 0 {
            return None;
        }
        Some(Self {
            iterator,
            current_entry: 0,
        })
    }
}

impl Drop for IoServiceIterator {
    fn drop(&mut self) {
        if self.current_entry != 0 {
            IOObjectRelease(self.current_entry);
        }
        IOObjectRelease(self.iterator);
    }
}

impl Iterator for IoServiceIterator {
    type Item = (io_registry_entry_t, String);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_entry != 0 {
            IOObjectRelease(self.current_entry);
            self.current_entry = 0;
        }
        let entry = IOIteratorNext(self.iterator);
        if entry == 0 {
            return None;
        }
        self.current_entry = entry;
        let mut name_buffer = [0 as c_char; 128];
        if unsafe { IORegistryEntryGetName(entry, &mut name_buffer) } != 0 {
            return None;
        }
        let name = unsafe { CStr::from_ptr(name_buffer.as_ptr()) }.to_string_lossy().into_owned();
        Some((entry, name))
    }
}
