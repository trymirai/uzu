implementing definitions;

// The comma-separated types for which a kernel's type parameter `name` is built, one entry point each.
[[__AttributeUsage(_AttributeTargets.Function)]]
struct VariantsAttribute {
    string name;
    string variants;
};

[[__AttributeUsage(_AttributeTargets.Function)]]
struct KernelAttribute;

[[__AttributeUsage(_AttributeTargets.Function)]]
struct PublicAttribute;

// The subgroup size, a host expression over the kernel's specializations, its pipelines require in full subgroups; other
// kernels run at whatever width the device chooses.
[[__AttributeUsage(_AttributeTargets.Function)]]
struct RequiredSubgroupSizeAttribute {
    string size;
};

// A kernel precondition the binding checks, a host expression: phase "new" over specializations, checked by the
// constructor before creating anything; phase "encode" also over uniforms, checked before recording.
[[__AttributeUsage(_AttributeTargets.Function)]]
struct PreconditionAttribute {
    string phase;
    string condition;
};

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
struct ConstantAttribute;

// A uniform canonical enum for which the binding builds one pipeline per enum value, passed as a specialization constant
// instead of a push constant; the kernel signature keeps it a uniform, chosen at encode.
[[__AttributeUsage(_AttributeTargets.Param)]]
struct PipelineVariantsAttribute;
