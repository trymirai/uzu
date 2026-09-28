#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <exception>
#include <memory>
#include <thread>

#include "engine.hpp"

namespace {

constexpr const char* kCapsuleName = "_splash_native.Engine";

struct Handle final {
    std::unique_ptr<benchmark::Engine> engine;
    std::thread::id owner = std::this_thread::get_id();
};

// Reacquire the GIL during stack unwinding before translating any C++ error.
class ReleaseGil final {
public:
    ReleaseGil() : state_(PyEval_SaveThread()) {
    }
    ~ReleaseGil() {
        PyEval_RestoreThread(state_);
    }
    ReleaseGil(const ReleaseGil&) = delete;
    ReleaseGil& operator=(const ReleaseGil&) = delete;

private:
    PyThreadState* state_;
};

PyObject* failure() {
    try {
        throw;
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown Splash native exception");
    }
    return nullptr;
}

Handle* handleFrom(
    PyObject* capsule,
    bool requireOpen = true
) {
    auto* handle = static_cast<Handle*>(PyCapsule_GetPointer(capsule, kCapsuleName));
    if (!handle)
        return nullptr;
    // Check the immutable owner before touching engine: close() destroys it
    // without the GIL, so a different Python thread must not read that pointer.
    if (handle->owner != std::this_thread::get_id()) {
        PyErr_SetString(PyExc_RuntimeError, "Splash engine must be used from the thread that created it");
        return nullptr;
    }
    if (requireOpen && !handle->engine) {
        PyErr_SetString(PyExc_RuntimeError, "Splash engine is closed");
        return nullptr;
    }
    return handle;
}

void destroyCapsule(PyObject* capsule) {
    auto* handle = static_cast<Handle*>(PyCapsule_GetPointer(capsule, kCapsuleName));
    if (!handle) {
        PyErr_Clear();
        return;
    }
    ReleaseGil released;
    delete handle;
}

PyObject* checkDevice(
    PyObject*,
    PyObject*
) {
    try {
        ReleaseGil released;
        benchmark::checkDevice();
    } catch (...) {
        return failure();
    }
    Py_RETURN_NONE;
}

PyObject* create(
    PyObject*,
    PyObject* args
) {
    const char* modelRoot;
    const char* metallibPath;
    if (!PyArg_ParseTuple(args, "ss:create", &modelRoot, &metallibPath))
        return nullptr;
    try {
        auto handle = std::make_unique<Handle>();
        {
            ReleaseGil released;
            handle->engine = std::make_unique<benchmark::Engine>(modelRoot, metallibPath);
        }
        PyObject* capsule = PyCapsule_New(handle.get(), kCapsuleName, destroyCapsule);
        if (!capsule) {
            ReleaseGil released;
            handle.reset();
            return nullptr;
        }
        static_cast<void>(handle.release());
        return capsule;
    } catch (...) {
        return failure();
    }
}

PyObject* asBytes(const std::vector<uint8_t>& bytes) {
    return PyBytes_FromStringAndSize(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<Py_ssize_t>(bytes.size())
    );
}

PyObject* receive(
    PyObject*,
    PyObject* args
) {
    PyObject* capsule;
    const char* data;
    Py_ssize_t size;
    if (!PyArg_ParseTuple(args, "Oy#:receive", &capsule, &data, &size))
        return nullptr;
    auto* handle = handleFrom(capsule);
    if (!handle)
        return nullptr;
    try {
        std::vector<uint8_t> output;
        {
            ReleaseGil released;
            output = handle->engine->receive({reinterpret_cast<const uint8_t*>(data), static_cast<size_t>(size)});
        }
        return asBytes(output);
    } catch (...) {
        return failure();
    }
}

PyObject* step(
    PyObject*,
    PyObject* args,
    PyObject* keywords
) {
    PyObject* capsule;
    double timeout = 0.1;
    static const char* names[] = {"handle", "timeout", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "O|d:step", names, &capsule, &timeout))
        return nullptr;
    auto* handle = handleFrom(capsule);
    if (!handle)
        return nullptr;
    try {
        std::vector<uint8_t> output;
        {
            ReleaseGil released;
            output = handle->engine->step(timeout);
        }
        return asBytes(output);
    } catch (...) {
        return failure();
    }
}

PyObject* status(
    PyObject*,
    PyObject* capsule
) {
    auto* handle = handleFrom(capsule);
    if (!handle)
        return nullptr;
    try {
        std::string output;
        {
            ReleaseGil released;
            output = handle->engine->status();
        }
        return PyUnicode_FromStringAndSize(output.data(), static_cast<Py_ssize_t>(output.size()));
    } catch (...) {
        return failure();
    }
}

PyObject* close(
    PyObject*,
    PyObject* capsule
) {
    auto* handle = handleFrom(capsule, false);
    if (!handle)
        return nullptr;
    {
        ReleaseGil released;
        handle->engine.reset();
    }
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"check_device", checkDevice, METH_NOARGS, "Validate the GPU using Splash's native device policy."},
    {"create", create, METH_VARARGS, "Load and warm the production runtime from a model root and metallib."},
    {"receive", receive, METH_VARARGS, "Feed protocol bytes directly to the native runtime and drain events."},
    {"step",
     _PyCFunction_CAST(step),
     METH_VARARGS | METH_KEYWORDS,
     "Drive native scheduling, waiting up to timeout seconds for events."},
    {"status", status, METH_O, "Return the production runtime status JSON."},
    {"close", close, METH_O, "Release native resources; safe to call again."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_splash_native",
    "In-process binding to Splash's production runtime.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

}  // namespace

PyMODINIT_FUNC PyInit__splash_native() {
    return PyModule_Create(&module);
}
