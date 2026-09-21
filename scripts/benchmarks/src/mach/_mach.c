#include <Python.h>

static struct PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "mach._mach",
    .m_doc = "Native macOS Mach memory collection.",
    .m_size = 0,
};

PyMODINIT_FUNC PyInit__mach(void) { return PyModule_Create(&module); }
