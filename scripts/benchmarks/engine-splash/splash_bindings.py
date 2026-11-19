# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-isysroot', '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 8
#
import ctypes; from install.paths import BINARY as library_path


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



_libraries = {}
_libraries['splash'] = ctypes.CDLL(library_path)
c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 8:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*8



try:
    benchmarks_splash_main = _libraries['splash'].benchmarks_splash_main
    benchmarks_splash_main.restype = ctypes.c_int32
    benchmarks_splash_main.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    benchmarks_splash_terminate = _libraries['splash'].benchmarks_splash_terminate
    benchmarks_splash_terminate.restype = None
    benchmarks_splash_terminate.argtypes = []
except AttributeError:
    pass
__all__ = \
    ['benchmarks_splash_main', 'benchmarks_splash_terminate']
