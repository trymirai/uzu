implementing definitions;

// Kernel annotations

[[__AttributeUsage(_AttributeTargets.Function)]]
struct VariantsAttribute {
    string name;
    string variants;
};

[[__AttributeUsage(_AttributeTargets.Function)]]
struct GroupSharedAttribute {
    string name;
    string size;
};

[[__AttributeUsage(_AttributeTargets.Function)]]
struct ConstraintAttribute {
    string constraint;
};

[[__AttributeUsage(_AttributeTargets.Function)]]
struct PublicAttribute;

[[__AttributeUsage(_AttributeTargets.Function)]]
struct KernelAttribute;

// The subgroup size, a host expression over the kernel's specializations, its pipelines require in full subgroups; other
// kernels run at whatever width the device chooses. Vulkan only.
[[__AttributeUsage(_AttributeTargets.Function)]]
struct RequiredSubgroupSizeAttribute {
    string size;
};

// A kernel precondition the binding checks, a host expression: phase "new" over specializations, checked by the
// constructor before creating anything; phase "encode" also over uniforms, checked before recording. Vulkan only.
[[__AttributeUsage(_AttributeTargets.Function)]]
struct PreconditionAttribute {
    string phase;
    string condition;
};

// Kernel argument annotation

[[__AttributeUsage(_AttributeTargets.Param)]]
struct ConstantAttribute;

[[__AttributeUsage(_AttributeTargets.Param)]]
struct OptionalAttribute {
    string condition;
};

[[__AttributeUsage(_AttributeTargets.Param)]]
struct SpecializeAttribute;

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

// A uniform canonical enum for which the binding builds one pipeline per enum value, passed as a specialization constant
// instead of a push constant; the kernel signature keeps it a uniform, chosen at encode. Vulkan only.
[[__AttributeUsage(_AttributeTargets.Param)]]
struct PipelineVariantsAttribute;
