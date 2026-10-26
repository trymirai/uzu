#!/usr/bin/env bash
set -euo pipefail

PYTHON_SUPPORT_URL="https://github.com/beeware/Python-Apple-support/releases/download/3.13-b16/Python-3.13-iOS-support.b16.tar.gz"
PYTHON_SUPPORT_SHA256="22b96f3d0c3663225f078556b036d2ee6ce3ff98c885706585240681335c71ea"
PYTHON_VERSION="3.13.16"
NATIVE_PACKAGES=(markupsafe numpy pydantic-core pyyaml regex safetensors tokenizers)
export IPHONEOS_DEPLOYMENT_TARGET="26.4"

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_DIR="$(cd -- "$APP_DIR/../../scripts/benchmarks" && pwd)"
DEPS_DIR="$APP_DIR/deps"
SLICE_DIR="$DEPS_DIR/Python.xcframework/ios-arm64"
VENV_DIR="$DEPS_DIR/venv"
WHEELS_DIR="$DEPS_DIR/wheels"
SOURCES_DIR="$DEPS_DIR/sources"
REQUIREMENTS="$DEPS_DIR/requirements.txt"

mkdir -p "$WHEELS_DIR" "$SOURCES_DIR"

if [[ ! -d "$DEPS_DIR/Python.xcframework" ]]; then
    curl -fsSL -o "$DEPS_DIR/python.tar.gz" "$PYTHON_SUPPORT_URL"
    echo "$PYTHON_SUPPORT_SHA256  $DEPS_DIR/python.tar.gz" | shasum -a 256 -c -
    tar -xzf "$DEPS_DIR/python.tar.gz" -C "$DEPS_DIR" Python.xcframework
    rm "$DEPS_DIR/python.tar.gz"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    uv venv --python 3.13 --seed "$VENV_DIR"
    "$VENV_DIR/bin/python" "$SLICE_DIR/platform-config/arm64-iphoneos/make_cross_venv.py" "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" --python-platform aarch64-apple-darwin \
        cmake cython maturin meson meson-python ninja pybind11 setuptools typing-extensions wheel
fi

cat > "$DEPS_DIR/ios.cmake" <<EOF
set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_SYSTEM_PROCESSOR arm64)
set(CMAKE_OSX_SYSROOT iphoneos)
set(CMAKE_OSX_ARCHITECTURES arm64)
set(CMAKE_OSX_DEPLOYMENT_TARGET $IPHONEOS_DEPLOYMENT_TARGET)
macro(set_xcode_property TARGET XCODE_PROPERTY XCODE_VALUE XCODE_VARIANT)
  set_property(TARGET \${TARGET} PROPERTY XCODE_ATTRIBUTE_\${XCODE_PROPERTY} \${XCODE_VALUE})
endmacro()
EOF

cat > "$DEPS_DIR/pyo3.txt" <<EOF
implementation=CPython
version=3.13
shared=true
lib_name=python3.13
lib_dir=$SLICE_DIR/lib
ext_suffix=.cpython-313-iphoneos.so
EOF

export PATH="$SLICE_DIR/bin:$VENV_DIR/bin:$PATH"
export CMAKE_TOOLCHAIN_FILE="$DEPS_DIR/ios.cmake"
export CMAKE_BUILD_PARALLEL_LEVEL="$(sysctl -n hw.ncpu)"
export CARGO_BUILD_TARGET="aarch64-apple-ios"
export PYO3_CONFIG_FILE="$DEPS_DIR/pyo3.txt"

uv export --project "$BENCHMARKS_DIR/engine-mlx" --frozen --no-dev --no-hashes --no-annotate --no-header \
    --no-emit-project --no-emit-local --no-emit-package hf-xet --quiet --output-file "$REQUIREMENTS"

pinned() {
    sed -n "s/^$1==\([^ ;]*\).*/\1/p" "$REQUIREMENTS"
}

build_wheel() {
    if ! compgen -G "$WHEELS_DIR/${1//-/_}-$2-*.whl" > /dev/null; then
        python -m pip wheel --no-build-isolation --no-deps --wheel-dir "$WHEELS_DIR" "$3"
    fi
}

for package in "${NATIVE_PACKAGES[@]}"; do
    version="$(pinned "$package")"
    build_wheel "$package" "$version" "$package==$version"
done

version="$(pinned sentencepiece)"
source_dir="$SOURCES_DIR/sentencepiece-$version"
if [[ ! -d "$source_dir" ]]; then
    curl -fsSL "https://files.pythonhosted.org/packages/source/s/sentencepiece/sentencepiece-$version.tar.gz" |
        tar -xz -C "$SOURCES_DIR"
    patch -d "$source_dir" -p1 < "$APP_DIR/patches/sentencepiece.patch"
fi
build_wheel sentencepiece "$version" "$source_dir"

version="$(pinned mlx)"
source_dir="$SOURCES_DIR/mlx-$version"
if [[ ! -d "$source_dir" ]]; then
    git clone --depth 1 --branch "v$version" https://github.com/ml-explore/mlx.git "$source_dir"
    patch -d "$source_dir" -p1 < "$APP_DIR/patches/mlx.patch"
fi
PYPI_RELEASE=1 \
    CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF -DMLX_BUILD_PYTHON_STUBS=OFF -DPython_INCLUDE_DIR=$SLICE_DIR/include/python3.13" \
    LDFLAGS="-F$SLICE_DIR -framework Python" \
    build_wheel mlx "$version" "$source_dir"

python -m pip wheel --no-build-isolation --no-deps --wheel-dir "$WHEELS_DIR" \
    "$BENCHMARKS_DIR/common-py" "$BENCHMARKS_DIR/engine-mlx"

rm -rf "$DEPS_DIR/app_packages"
uv pip install --target "$DEPS_DIR/app_packages" --python 3.13 --python-platform arm64-apple-ios --python-version "$PYTHON_VERSION" \
    --only-binary :all: --no-deps --compile-bytecode --find-links "$WHEELS_DIR" --requirements "$REQUIREMENTS" \
    "$WHEELS_DIR"/benchmarks_common-*.whl "$WHEELS_DIR"/benchmarks_mlx-*.whl
rm -rf "$DEPS_DIR/app_packages/mlx/lib/libmlx.a" "$DEPS_DIR/app_packages/mlx/include" "$DEPS_DIR/app_packages/mlx/share"
