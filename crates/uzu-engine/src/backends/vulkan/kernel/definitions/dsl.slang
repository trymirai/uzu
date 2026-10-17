implementing definitions;

// Kernel annotations

[[__AttributeUsage(_AttributeTargets.Function)]]
struct KernelAttribute;

[[__AttributeUsage(_AttributeTargets.Function)]]
struct PublicAttribute;

// The subgroup size, a host expression over the kernel's specializations, its pipelines require in full subgroups
// instead of the device's policy.
[[__AttributeUsage(_AttributeTargets.Function)]]
struct RequiredSubgroupSizeAttribute {
    string size;
};


// Kernel parameter annotation

[[__AttributeUsage(_AttributeTargets.Param)]]
struct AxisAttribute {
    string threads;
    string threads_in_group;
};

[[__AttributeUsage(_AttributeTargets.Param)]]
struct GroupsAttribute {
    string groups;
};

[[__AttributeUsage(_AttributeTargets.Param)]]
struct ThreadsAttribute {
    string threads;
};

[[__AttributeUsage(_AttributeTargets.Param)]]
struct OptionalAttribute {
    string condition;
};

[[__AttributeUsage(_AttributeTargets.Param)]]
struct SpecializeAttribute;

// A read-only pointer to a canonical GPU struct that the binding takes as a host slice, like the CPU kernel, and uploads
// when it records the dispatch.
[[__AttributeUsage(_AttributeTargets.Param)]]
struct HostSliceAttribute;

// A uniform canonical enum for which the binding builds one pipeline per enum value, passed as a specialization constant
// instead of a push constant; the kernel signature keeps it a uniform, chosen at encode.
[[__AttributeUsage(_AttributeTargets.Param)]]
struct PipelineVariantsAttribute;
