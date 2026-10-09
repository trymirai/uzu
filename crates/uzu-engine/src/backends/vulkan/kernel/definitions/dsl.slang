implementing definitions;

// Specialization constant annotations

[[__AttributeUsage(_AttributeTargets.Var)]]
struct ThreadgroupSizeAttribute;

[[__AttributeUsage(_AttributeTargets.Var)]]
struct SimdgroupSizeAttribute;


// Kernel annotations

[[__AttributeUsage(_AttributeTargets.Function)]]
struct KernelAttribute;

[[__AttributeUsage(_AttributeTargets.Function)]]
struct PublicAttribute;

// Private kernel that still gets a generated binding, compiled only into test builds.
[[__AttributeUsage(_AttributeTargets.Function)]]
struct TestAttribute;


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
// when it records the dispatch. With a positive length N, a read-only uint pointer the binding takes as the CPU's
// &[u32; N].
[[__AttributeUsage(_AttributeTargets.Param)]]
struct HostSliceAttribute {
    int length = 0;
};

// A uniform canonical enum for which the binding builds one pipeline per enum value, passed as a specialization constant
// instead of a push constant; the kernel signature keeps it a uniform, chosen at encode.
[[__AttributeUsage(_AttributeTargets.Param)]]
struct PipelineVariantsAttribute;
