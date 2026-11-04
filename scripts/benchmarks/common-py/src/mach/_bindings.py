# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-isysroot', '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 8
#
import ctypes; from importlib.util import find_spec


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                type_ = type_._type_
                if hasattr(type_, 'as_dict'):
                    value = [type_.as_dict(v) for v in value]
                else:
                    value = [i for i in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



_libraries = {}
_libraries['_native.cpython-313-darwin.so'] = ctypes.CDLL(find_spec(__package__ + '._native').origin)
def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))





class struct_memory_counters_t(Structure):
    pass

struct_memory_counters_t._pack_ = 1 # source:False
struct_memory_counters_t._fields_ = [
    ('pid', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('phys_footprint', ctypes.c_uint64),
    ('resident_size', ctypes.c_uint64),
    ('resident_size_peak', ctypes.c_uint64),
    ('device', ctypes.c_uint64),
    ('device_peak', ctypes.c_uint64),
    ('internal', ctypes.c_uint64),
    ('compressed', ctypes.c_uint64),
    ('graphics_footprint', ctypes.c_uint64),
    ('graphics_footprint_compressed', ctypes.c_uint64),
    ('graphics_nofootprint', ctypes.c_uint64),
    ('graphics_nofootprint_compressed', ctypes.c_uint64),
    ('graphics_total', ctypes.c_uint64),
    ('malloc_allocated', ctypes.c_uint64),
    ('malloc_in_use', ctypes.c_uint64),
    ('malloc_max_in_use', ctypes.c_uint64),
]

memory_counters_t = struct_memory_counters_t
kern_return_t = ctypes.c_int32
try:
    get_memory_counters = _libraries['_native.cpython-313-darwin.so'].get_memory_counters
    get_memory_counters.restype = kern_return_t
    get_memory_counters.argtypes = [ctypes.POINTER(struct_memory_counters_t), ctypes.c_bool]
except AttributeError:
    pass
int32_t = ctypes.c_int32
try:
    get_memory_counters_for_pid = _libraries['_native.cpython-313-darwin.so'].get_memory_counters_for_pid
    get_memory_counters_for_pid.restype = kern_return_t
    get_memory_counters_for_pid.argtypes = [ctypes.POINTER(struct_memory_counters_t), int32_t, ctypes.c_bool]
except AttributeError:
    pass
try:
    memory_counters_error_string = _libraries['_native.cpython-313-darwin.so'].memory_counters_error_string
    memory_counters_error_string.restype = ctypes.POINTER(ctypes.c_char)
    memory_counters_error_string.argtypes = [kern_return_t]
except AttributeError:
    pass
class struct_apple_temp_sensors_t(Structure):
    pass

struct_apple_temp_sensors_t._pack_ = 1 # source:False
struct_apple_temp_sensors_t._fields_ = [
    ('cpu_avg', ctypes.c_double),
    ('gpu_avg', ctypes.c_double),
]

apple_temp_sensors_t = struct_apple_temp_sensors_t
try:
    get_temp_sensors = _libraries['_native.cpython-313-darwin.so'].get_temp_sensors
    get_temp_sensors.restype = kern_return_t
    get_temp_sensors.argtypes = [ctypes.POINTER(struct_apple_temp_sensors_t)]
except AttributeError:
    pass
try:
    await_cooldown = _libraries['_native.cpython-313-darwin.so'].await_cooldown
    await_cooldown.restype = kern_return_t
    await_cooldown.argtypes = [ctypes.c_double, ctypes.c_bool]
except AttributeError:
    pass
try:
    get_gpu_energy = _libraries['_native.cpython-313-darwin.so'].get_gpu_energy
    get_gpu_energy.restype = kern_return_t
    get_gpu_energy.argtypes = [ctypes.POINTER(ctypes.c_double)]
except AttributeError:
    pass
__all__ = \
    ['apple_temp_sensors_t', 'await_cooldown', 'get_gpu_energy',
    'get_memory_counters', 'get_memory_counters_for_pid',
    'get_temp_sensors', 'int32_t', 'kern_return_t',
    'memory_counters_error_string', 'memory_counters_t',
    'struct_apple_temp_sensors_t', 'struct_memory_counters_t']
