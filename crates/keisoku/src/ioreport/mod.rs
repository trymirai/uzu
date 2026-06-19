use core::ffi::c_void;

mod report;
mod sample;

pub use report::IoReport;
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
