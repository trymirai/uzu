#include "energy_counters.h"

#include <CoreFoundation/CoreFoundation.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>

typedef struct IOReportSubscription* IOReportSubscriptionRef;

CFMutableDictionaryRef IOReportCopyChannelsInGroup(
    CFStringRef,
    CFStringRef,
    uint64_t,
    uint64_t,
    uint64_t
);
IOReportSubscriptionRef IOReportCreateSubscription(
    void*,
    CFMutableDictionaryRef,
    CFMutableDictionaryRef*,
    uint64_t,
    CFTypeRef
);
CFDictionaryRef IOReportCreateSamples(
    IOReportSubscriptionRef,
    CFMutableDictionaryRef,
    CFTypeRef
);
CFStringRef IOReportChannelGetChannelName(CFDictionaryRef);
uint64_t IOReportChannelGetUnit(CFDictionaryRef);
int64_t IOReportSimpleGetIntegerValue(
    CFDictionaryRef,
    int32_t
);

static pthread_once_t subscribe_once = PTHREAD_ONCE_INIT;
static IOReportSubscriptionRef subscription = NULL;
static CFMutableDictionaryRef subscribed_channels = NULL;

static void subscribe(void) {
    CFMutableDictionaryRef group = IOReportCopyChannelsInGroup(CFSTR("Energy Model"), NULL, 0, 0, 0);
    if (group == NULL) {
        return;
    }
    CFArrayRef channels = CFDictionaryGetValue(group, CFSTR("IOReportChannels"));
    CFMutableArrayRef selected = CFArrayCreateMutable(NULL, 1, &kCFTypeArrayCallBacks);
    for (CFIndex index = 0; channels != NULL && index < CFArrayGetCount(channels); index++) {
        CFDictionaryRef channel = CFArrayGetValueAtIndex(channels, index);
        CFStringRef name = IOReportChannelGetChannelName(channel);
        if (name != NULL && CFStringCompare(name, CFSTR("GPU Energy"), 0) == kCFCompareEqualTo) {
            CFArrayAppendValue(selected, channel);
        }
    }
    if (CFArrayGetCount(selected) == 1) {
        CFDictionarySetValue(group, CFSTR("IOReportChannels"), selected);
        subscription = IOReportCreateSubscription(NULL, group, &subscribed_channels, 0, NULL);
    }
    CFRelease(selected);
    CFRelease(group);
}

kern_return_t get_gpu_energy(double* joules) {
    pthread_once(&subscribe_once, subscribe);
    if (subscription == NULL) {
        return KERN_FAILURE;
    }

    CFDictionaryRef samples = IOReportCreateSamples(subscription, subscribed_channels, NULL);
    if (samples == NULL) {
        return KERN_FAILURE;
    }

    kern_return_t result = KERN_FAILURE;
    CFArrayRef channels = CFDictionaryGetValue(samples, CFSTR("IOReportChannels"));
    if (channels != NULL && CFArrayGetCount(channels) == 1) {
        CFDictionaryRef channel = CFArrayGetValueAtIndex(channels, 0);
        // The unit encodes the quantity in bits 56-63 (3 is energy) and a biased SI exponent in bits 32-39.
        const uint64_t unit = IOReportChannelGetUnit(channel);
        if ((unit >> 56 & 0xff) == 3) {
            *joules = (double)IOReportSimpleGetIntegerValue(channel, 0) * pow(10, (int)(unit >> 32 & 0xff) - 127);
            result = KERN_SUCCESS;
        }
    }
    CFRelease(samples);
    return result;
}
