implementing definitions;

// Specialization constant annotations

[[__AttributeUsage(_AttributeTargets.Var)]]
struct ThreadgroupSizeAttribute;

[[__AttributeUsage(_AttributeTargets.Var)]]
struct SimdgroupSizeAttribute;


// Kernel annotations

[[__AttributeUsage(_AttributeTargets.Function)]]
struct KernelAttribute;


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
