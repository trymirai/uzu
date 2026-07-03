use core::{ffi::c_void, ptr::NonNull};

use objc2_core_foundation::{CFDictionary, CFMutableDictionary, CFRetained, CFString, CFType, Type};

mod bandwidth;
mod channel;
mod energy_totals;
mod frequency;
mod report;
mod residency;
mod sample;
mod subscription;

pub use report::IoReport;
pub(crate) use report::RawEnergySample;
pub use sample::SocSample;

kanka::ffi_table! {
    struct IoReportFunctions from "/usr/lib/libIOReport.dylib" {
        copy_channels_in_group = "IOReportCopyChannelsInGroup":
            unsafe extern "C" fn(*const c_void, *const c_void, u64, u64, u64) -> *const c_void,
        merge_channels = "IOReportMergeChannels":
            unsafe extern "C" fn(*const c_void, *const c_void, *const c_void),
        create_subscription = "IOReportCreateSubscription":
            unsafe extern "C" fn(*const c_void, *mut c_void, *mut *mut c_void, u64, *const c_void) -> *const c_void,
        create_samples = "IOReportCreateSamples":
            unsafe extern "C" fn(*const c_void, *mut c_void, *const c_void) -> *const c_void,
        create_samples_delta = "IOReportCreateSamplesDelta":
            unsafe extern "C" fn(*const c_void, *const c_void, *const c_void) -> *const c_void,
        channel_get_group = "IOReportChannelGetGroup":
            unsafe extern "C" fn(*const c_void) -> *const c_void,
        channel_get_subgroup = "IOReportChannelGetSubGroup":
            unsafe extern "C" fn(*const c_void) -> *const c_void,
        channel_get_channel_name = "IOReportChannelGetChannelName":
            unsafe extern "C" fn(*const c_void) -> *const c_void,
        channel_get_unit_label = "IOReportChannelGetUnitLabel":
            unsafe extern "C" fn(*const c_void) -> *const c_void,
        simple_get_integer_value = "IOReportSimpleGetIntegerValue":
            unsafe extern "C" fn(*const c_void, i32) -> i64,
        state_get_count = "IOReportStateGetCount":
            unsafe extern "C" fn(*const c_void) -> i32,
        state_get_name_for_index = "IOReportStateGetNameForIndex":
            unsafe extern "C" fn(*const c_void, i32) -> *const c_void,
        state_get_residency = "IOReportStateGetResidency":
            unsafe extern "C" fn(*const c_void, i32) -> i64,
    }
}

impl IoReportFunctions {
    fn copy_channels_in_group(
        &self,
        group: &CFString,
        subgroup: Option<&CFString>,
    ) -> Option<CFRetained<CFMutableDictionary>> {
        let subgroup_pointer = subgroup.map_or(core::ptr::null_mut(), raw);
        let channels = unsafe { (self.copy_channels_in_group)(raw(group), subgroup_pointer, 0, 0, 0) };
        retained_mutable_dictionary(channels)
    }

    fn merge_channels(
        &self,
        first: &CFMutableDictionary,
        other: &CFMutableDictionary,
    ) {
        unsafe { (self.merge_channels)(raw(first), raw(other), core::ptr::null()) };
    }

    fn create_subscription(
        &self,
        channels: &CFMutableDictionary,
    ) -> Option<(CFRetained<CFType>, Option<CFRetained<CFDictionary>>)> {
        let mut subscribed: *mut c_void = core::ptr::null_mut();
        let subscription = unsafe {
            (self.create_subscription)(core::ptr::null(), raw(channels), &mut subscribed, 0, core::ptr::null())
        };
        let subscription = unsafe { CFRetained::from_raw(NonNull::new(subscription.cast_mut().cast::<CFType>())?) };
        Some((subscription, retained_dictionary(subscribed)))
    }

    fn create_samples(
        &self,
        subscription: &CFType,
        channels: &CFMutableDictionary,
    ) -> Option<CFRetained<CFDictionary>> {
        let samples = unsafe { (self.create_samples)(raw(subscription), raw(channels), core::ptr::null()) };
        retained_dictionary(samples)
    }

    fn create_samples_delta(
        &self,
        previous: &CFDictionary,
        next: &CFDictionary,
    ) -> Option<CFRetained<CFDictionary>> {
        let delta = unsafe { (self.create_samples_delta)(raw(previous), raw(next), core::ptr::null()) };
        retained_dictionary(delta)
    }
}

fn raw<T: Type>(value: &T) -> *mut c_void {
    let pointer: *const T = value;
    pointer.cast::<c_void>().cast_mut()
}

fn retained_dictionary(value: *const c_void) -> Option<CFRetained<CFDictionary>> {
    NonNull::new(value.cast_mut().cast::<CFDictionary>()).map(|pointer| unsafe { CFRetained::from_raw(pointer) })
}

fn retained_mutable_dictionary(value: *const c_void) -> Option<CFRetained<CFMutableDictionary>> {
    NonNull::new(value.cast_mut().cast::<CFMutableDictionary>()).map(|pointer| unsafe { CFRetained::from_raw(pointer) })
}
