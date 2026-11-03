import Foundation
import os

enum PythonRuntime {
    private static let environment = OSAllocatedUnfairLock<String?>(initialState: nil)

    @discardableResult
    private static func start(_ environment: String) -> String {
        Self.environment.withLock { running in
            if running == nil {
                DispatchQueue.main.sync {
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
                        site.addsitedir(os.path.join(bundle, "envs", "\(environment)"))
                        """,
                        nil
                    )
                    PyEval_SaveThread()
                }
                running = environment
            }
            return running!
        }
    }

    static func main(_ environment: String, _ module: String, _ arguments: [String]) -> Int32 {
        let running = start(environment)
        guard running == environment else {
            fputs(
                "Python runs the \(running) environment; relaunch the app to run \(environment)\n",
                stderr)
            return 1
        }
        return call("main", [module] + arguments) { Int32(PyLong_AsLong($0)) }
    }

    static func modelPaths(_ arguments: [String]) -> [String] {
        start("mlx")
        return call("model_paths", arguments) { result in
            (0..<PyList_Size(result)).map {
                String(cString: PyUnicode_AsUTF8(PyList_GetItem(result, $0)))
            }
        }
    }

    private static func call<Result>(
        _ name: String, _ values: [String], _ convert: (UnsafeMutablePointer<PyObject>) -> Result
    ) -> Result {
        let state = PyGILState_Ensure()
        defer { PyGILState_Release(state) }
        guard let runner = PyImport_ImportModule("runner") else {
            PyErr_Print()
            fatalError("Cannot import runner")
        }
        let function = PyObject_GetAttrString(runner, name)
        let tuple = PyTuple_New(values.count)
        for (index, value) in values.enumerated() {
            PyTuple_SetItem(tuple, index, PyUnicode_FromString(value))
        }
        let result = PyObject_CallObject(function, tuple)
        defer { [result, tuple, function, runner].forEach(Py_DecRef) }
        guard let result else {
            PyErr_Print()
            fatalError("runner.\(name) failed")
        }
        return convert(result)
    }

    private static func check(_ status: PyStatus) {
        if PyStatus_Exception(status) != 0 {
            Py_ExitStatusException(status)
        }
    }
}
