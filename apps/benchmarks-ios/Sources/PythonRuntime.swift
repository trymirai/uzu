import Foundation

enum PythonRuntime {
    static func start() {
        let fileManager = FileManager.default
        setenv("HOME", fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0].path, 1)
        fileManager.changeCurrentDirectoryPath(
            fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0].path)

        var preconfig = PyPreConfig()
        PyPreConfig_InitIsolatedConfig(&preconfig)
        preconfig.utf8_mode = 1
        check(Py_PreInitialize(&preconfig))

        let config = UnsafeMutablePointer<PyConfig>.allocate(capacity: 1)
        defer { config.deallocate() }
        PyConfig_InitIsolatedConfig(config)
        config.pointee.write_bytecode = 0
        let home = "\(Bundle.main.resourcePath!)/python"
        check(PyConfig_SetBytesString(config, &config.pointee.home, home))
        check(Py_InitializeFromConfig(config))
        PyConfig_Clear(config)
        PyRun_SimpleStringFlags(
            """
            import os, site, sys
            bundle = os.path.dirname(sys.executable)
            sys.path.insert(0, os.path.join(bundle, "app"))
            site.addsitedir(os.path.join(bundle, "app_packages"))
            """,
            nil
        )
        PyEval_SaveThread()
    }

    static func run(engine: String, model: String, requests: String) -> String {
        let state = PyGILState_Ensure()
        defer { PyGILState_Release(state) }
        guard let module = PyImport_ImportModule("bench_ios") else {
            PyErr_Print()
            fatalError("Cannot import bench_ios")
        }
        let function = PyObject_GetAttrString(module, "run")
        let arguments = PyTuple_New(3)
        PyTuple_SetItem(arguments, 0, PyUnicode_FromString(engine))
        PyTuple_SetItem(arguments, 1, PyUnicode_FromString(model))
        PyTuple_SetItem(arguments, 2, PyUnicode_FromString(requests))
        let result = PyObject_CallObject(function, arguments)
        defer { [result, arguments, function, module].forEach(Py_DecRef) }
        guard let output = PyUnicode_AsUTF8(result) else {
            PyErr_Print()
            fatalError("bench_ios.run failed")
        }
        return String(cString: output)
    }

    private static func check(_ status: PyStatus) {
        if PyStatus_Exception(status) != 0 {
            Py_ExitStatusException(status)
        }
    }
}
