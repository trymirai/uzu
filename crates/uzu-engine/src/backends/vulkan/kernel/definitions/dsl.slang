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
