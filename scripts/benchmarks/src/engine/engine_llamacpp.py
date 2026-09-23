import os
import subprocess
from pathlib import Path

from bench import BenchResponse

project_dir = Path(__file__).resolve().parents[2]
build_dir = project_dir / "build" / "llamacpp-release"


def _build():
    # create config
    result = subprocess.run(
        [
            "cmake",
            "-S",
            "src/llamacpp",
            "-B",
            "build/llamacpp-release",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={build_dir}",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DGGML_LTO=ON",
            "-DGGML_NATIVE=ON",
            "-DGGML_METAL=ON",
            "-DGGML_METAL_EMBED_LIBRARY=ON",
            "-DGGML_METAL_NDEBUG=ON",
            "-DGGML_METAL_SHADER_DEBUG=OFF",
            "-DGGML_ACCELERATE=ON",
            "-DGGML_BLAS=ON",
            "-DGGML_BLAS_VENDOR=Apple",
        ],
        cwd=project_dir,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        if result.stderr:
            raise RuntimeError(result.stderr)
        raise RuntimeError(f"CMake configuration failed with exit code {result.returncode}")

    # build
    result = subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--parallel",
            str(os.cpu_count() or 1),
        ],
        cwd=project_dir,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        if result.stderr:
            raise RuntimeError(result.stderr)
        raise RuntimeError(f"CMake build failed with exit code {result.returncode}")


def _execute(model: str | Path, input_path: Path) -> str:
    executable = build_dir / "llamacpp_bench"
    result = subprocess.run(
        [
            str(executable),
            "--model",
            model,
            "--input",
            input_path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if result.stderr:
            raise RuntimeError(result.stderr)
        raise RuntimeError(f"llama.cpp execution failed with exit code {result.returncode}")
    return result.stdout


def run(model: str | Path, input_path: Path) -> BenchResponse:
    _build()
    output_json = _execute(model, input_path)
    return BenchResponse.model_validate_json(output_json)
