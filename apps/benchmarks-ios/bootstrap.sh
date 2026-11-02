#!/usr/bin/env bash
set -euo pipefail

PYTHON_SUPPORT_URL="https://github.com/beeware/Python-Apple-support/releases/download/3.13-b16/Python-3.13-iOS-support.b16.tar.gz"
PYTHON_SUPPORT_SHA256="22b96f3d0c3663225f078556b036d2ee6ce3ff98c885706585240681335c71ea"
PYTHON_VERSION="3.13.16"
OPENSSL_URL="https://github.com/beeware/cpython-apple-source-deps/releases/download/OpenSSL-3.5.9-1/openssl-3.5.9-1-iphoneos.arm64.tar.gz"
OPENSSL_SHA256="aa8c3a96138a5650f51a1416076f947fec9ae7a5fba83afb0015fefe314107dc"
NATIVE_PACKAGES=(llguidance markupsafe numpy openai-harmony pydantic-core pyyaml regex rpds-py safetensors tokenizers)
OMLX_UNUSED_PACKAGES=(cohere-melody cryptography llguidance lxml miniaudio onnxruntime opencv-python primp psutil
    pypdfium2 scipy setproctitle tiktoken)
export IPHONEOS_DEPLOYMENT_TARGET="26.4"

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$APP_DIR/../.." && pwd)"
BENCHMARKS_DIR="$REPO_DIR/scripts/benchmarks"
DEPS_DIR="$APP_DIR/deps"
SLICE_DIR="$DEPS_DIR/Python.xcframework/ios-arm64"
VENV_DIR="$DEPS_DIR/venv"
WHEELS_DIR="$DEPS_DIR/wheels"
SOURCES_DIR="$DEPS_DIR/sources"

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
fi
uv pip install --quiet --python "$VENV_DIR/bin/python" --python-platform aarch64-apple-darwin \
    cffi cmake cython maturin meson meson-python ninja pybind11 setuptools typing-extensions wheel

if [[ ! -d "$DEPS_DIR/openssl" ]]; then
    curl -fsSL -o "$DEPS_DIR/openssl.tar.gz" "$OPENSSL_URL"
    echo "$OPENSSL_SHA256  $DEPS_DIR/openssl.tar.gz" | shasum -a 256 -c -
    mkdir -p "$DEPS_DIR/openssl"
    tar -xzf "$DEPS_DIR/openssl.tar.gz" -C "$DEPS_DIR/openssl"
    rm "$DEPS_DIR/openssl.tar.gz"
fi

framework() {
    FRAMEWORK_DIR="$DEPS_DIR/Benchmarks$1.framework"
    rm -rf "$FRAMEWORK_DIR"
    mkdir -p "$FRAMEWORK_DIR"
    cp "$DEPS_DIR/Python.xcframework/build/iOS-dylib-Info-template.plist" "$FRAMEWORK_DIR/Info.plist"
    plutil -replace CFBundleExecutable -string "Benchmarks$1" "$FRAMEWORK_DIR/Info.plist"
    plutil -replace CFBundleIdentifier -string "com.mirai.tech.benchmarks.$2" "$FRAMEWORK_DIR/Info.plist"
}

(cd "$REPO_DIR" && cargo rustc --release --target aarch64-apple-ios -p benchmarks-uzu --lib --crate-type cdylib)
framework Uzu uzu
cp "$REPO_DIR/target/aarch64-apple-ios/release/libbenchmarks_uzu.dylib" "$FRAMEWORK_DIR/BenchmarksUzu"
install_name_tool -id @rpath/BenchmarksUzu.framework/BenchmarksUzu "$FRAMEWORK_DIR/BenchmarksUzu"

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

export CMAKE_BUILD_PARALLEL_LEVEL="$(sysctl -n hw.ncpu)"
LLAMACPP_BUILD_DIR="$DEPS_DIR/llamacpp-build"
cmake -S "$BENCHMARKS_DIR/engine-llamacpp" -B "$LLAMACPP_BUILD_DIR" -G "Unix Makefiles" \
    -DCMAKE_TOOLCHAIN_FILE="$DEPS_DIR/ios.cmake" -DCMAKE_BUILD_TYPE=Release -DCMAKE_MACOSX_BUNDLE=OFF \
    -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_METAL_NDEBUG=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple \
    -DLLAMA_BUILD_BORINGSSL=ON >/dev/null
cmake --build "$LLAMACPP_BUILD_DIR" --target engine-llamacpp >/dev/null
framework Llamacpp llamacpp
cp "$LLAMACPP_BUILD_DIR/libBenchmarksLlamacpp.dylib" "$FRAMEWORK_DIR/BenchmarksLlamacpp"
install_name_tool -id @rpath/BenchmarksLlamacpp.framework/BenchmarksLlamacpp "$FRAMEWORK_DIR/BenchmarksLlamacpp"

MLDRIFT_DIR="$BENCHMARKS_DIR/engine-mldrift"
"$MLDRIFT_DIR/bootstrap.sh"
(cd "$MLDRIFT_DIR/deps/ml-drift" && bazelisk --output_base="$DEPS_DIR/mldrift-bazel" build -c opt \
    --platforms=@build_bazel_apple_support//platforms:ios_arm64 --ios_minimum_os="$IPHONEOS_DEPLOYMENT_TARGET" \
    --copt=-DABSL_FLAGS_STRIP_NAMES=0 --symlink_prefix=bazel-ios- //benchmarks:BenchmarksMldrift >&2)
framework Mldrift mldrift
cp "$MLDRIFT_DIR/deps/ml-drift/bazel-ios-bin/benchmarks/libBenchmarksMldrift.dylib" "$FRAMEWORK_DIR/BenchmarksMldrift"
chmod u+w "$FRAMEWORK_DIR/BenchmarksMldrift"
install_name_tool -id @rpath/BenchmarksMldrift.framework/BenchmarksMldrift "$FRAMEWORK_DIR/BenchmarksMldrift"

SPLASH_DIR="$BENCHMARKS_DIR/engine-splash"
SPLASH_SOURCE_DIR="$SPLASH_DIR/deps/splash"
"$SPLASH_DIR/bootstrap.sh"
make -C "$SPLASH_SOURCE_DIR" --no-print-directory -j "$CMAKE_BUILD_PARALLEL_LEVEL" BUILD=build-ios \
    BUILD_ID_PYTHON="uv run --no-project python" \
    CXX="xcrun -sdk iphoneos clang++ -target arm64-apple-ios$IPHONEOS_DEPLOYMENT_TARGET" \
    METAL="xcrun -sdk iphoneos metal" METALLIB="xcrun -sdk iphoneos metallib" AR="xcrun -sdk iphoneos ar" \
    MACOS_TARGET_FLAG="-mios-version-min=$IPHONEOS_DEPLOYMENT_TARGET" \
    build-ios/engine/libsplash.a build-ios/engine/main.o build-ios/splash.metallib >&2
framework Splash splash
xcrun --sdk iphoneos clang++ -target "arm64-apple-ios$IPHONEOS_DEPLOYMENT_TARGET" -dynamiclib -fobjc-arc \
    -o "$FRAMEWORK_DIR/BenchmarksSplash" -install_name @rpath/BenchmarksSplash.framework/BenchmarksSplash \
    "$SPLASH_SOURCE_DIR/build-ios/engine/main.o" "$SPLASH_SOURCE_DIR/build-ios/engine/libsplash.a" \
    -framework Foundation -framework IOKit -framework IOSurface -framework Metal \
    -Wl,-exported_symbol,_benchmarks_splash_main -Wl,-exported_symbol,_benchmarks_splash_terminate
cp "$SPLASH_SOURCE_DIR/build-ios/splash.metallib" "$FRAMEWORK_DIR/"

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
export CARGO_BUILD_TARGET="aarch64-apple-ios"
export PYO3_CONFIG_FILE="$DEPS_DIR/pyo3.txt"
export CARGO_TARGET_AARCH64_APPLE_IOS_RUSTFLAGS="-C link-arg=-F$SLICE_DIR -C link-arg=-framework -C link-arg=Python"

requirements() {
    uv export --project "$BENCHMARKS_DIR/engine-$1" --frozen --no-dev --no-hashes --no-annotate --no-header \
        --no-emit-project --no-emit-local --no-emit-package hf-xet --no-emit-package mlx-metal "${@:2}" --quiet |
        sed "s/ ; platform_machine == 'arm64' and sys_platform == 'darwin'$//"
}

pinned() {
    sed -n "s/^$1==\([^ ;]*\).*/\1/p" "$DEPS_DIR/requirements-$2.txt"
}

build_wheel() {
    if ! compgen -G "$WHEELS_DIR/${1//-/_}-$2-*.whl" > /dev/null; then
        RUSTUP_TOOLCHAIN=stable python -m pip wheel --no-build-isolation --no-deps --wheel-dir "$WHEELS_DIR" "$3"
    fi
}

mlx_source() {
    local source_dir="$SOURCES_DIR/mlx-$1"
    if [[ ! -d "$source_dir" ]]; then
        git clone --depth 1 --branch "v$1" https://github.com/ml-explore/mlx.git "$source_dir" >&2
        patch -d "$source_dir" -p1 < "$APP_DIR/patches/mlx.patch" >&2
    fi
    echo "$source_dir"
}

build_native_wheels() {
    for package in "${NATIVE_PACKAGES[@]}"; do
        version="$(pinned "$package" "$1")"
        if [[ -n "$version" ]]; then
            build_wheel "$package" "$version" "$package==$version"
        fi
    done

    version="$(pinned sentencepiece "$1")"
    if [[ -n "$version" ]]; then
        source_dir="$SOURCES_DIR/sentencepiece-$version"
        if [[ ! -d "$source_dir" ]]; then
            curl -fsSL "https://files.pythonhosted.org/packages/source/s/sentencepiece/sentencepiece-$version.tar.gz" |
                tar -xz -C "$SOURCES_DIR"
            patch -d "$source_dir" -p1 < "$APP_DIR/patches/sentencepiece.patch"
        fi
        build_wheel sentencepiece "$version" "$source_dir"
    fi

    version="$(pinned cryptography "$1")"
    if [[ -n "$version" ]]; then
        OPENSSL_DIR="$DEPS_DIR/openssl" build_wheel cryptography "$version" "cryptography==$version"
    fi

    version="$(pinned mlx "$1")"
    if [[ -n "$version" ]]; then
        PYPI_RELEASE=1 \
            CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF -DMLX_BUILD_PYTHON_STUBS=OFF -DPython_INCLUDE_DIR=$SLICE_DIR/include/python3.13" \
            LDFLAGS="-F$SLICE_DIR -framework Python" \
            build_wheel mlx "$version" "$(mlx_source "$version")"
    fi
}

requirements mlx > "$DEPS_DIR/requirements-mlx.txt"
requirements omlx "${OMLX_UNUSED_PACKAGES[@]/#/--no-emit-package=}" > "$DEPS_DIR/requirements-omlx.txt"
requirements mtplx > "$DEPS_DIR/requirements-mtplx.txt"
requirements splash > "$DEPS_DIR/requirements-splash.txt"

for environment in mlx omlx mtplx splash; do
    build_native_wheels "$environment"
done

rm -f "$WHEELS_DIR"/benchmarks_*.whl
python -m pip wheel --no-build-isolation --no-deps --wheel-dir "$WHEELS_DIR" "$BENCHMARKS_DIR/common-py" \
    "$BENCHMARKS_DIR/engine-mlx" "$BENCHMARKS_DIR/engine-omlx" "$BENCHMARKS_DIR/engine-mtplx" \
    "$BENCHMARKS_DIR/engine-splash"

rm -rf "$DEPS_DIR/envs"
for environment in mlx omlx mtplx splash; do
    target="$DEPS_DIR/envs/$environment"
    uv pip install --target "$target" --python 3.13 --python-platform arm64-apple-ios \
        --python-version "$PYTHON_VERSION" --only-binary :all: --no-deps --compile-bytecode --find-links "$WHEELS_DIR" \
        --requirements "$DEPS_DIR/requirements-$environment.txt" \
        "$WHEELS_DIR"/benchmarks_common-*.whl "$WHEELS_DIR"/benchmarks_$environment-*.whl
    rm -rf "$target/mlx/lib/libmlx.a" "$target/mlx/include" "$target/mlx/share"
done
for tree in install server; do
    rsync -a --exclude __pycache__ --exclude models "$SPLASH_SOURCE_DIR/$tree" "$DEPS_DIR/envs/splash/deps/splash/"
done

MLXSERVE_DIR="$BENCHMARKS_DIR/engine-mlxserve"
MLXSERVE_SOURCE_DIR="$MLXSERVE_DIR/deps/mlx-serve"
MLXSERVE_BUILD_DIR="$DEPS_DIR/mlxserve-build"
MLXSERVE_STAGE_DIR="$MLXSERVE_BUILD_DIR/stage"
"$MLXSERVE_DIR/bootstrap.sh"
version="$(sed -En 's/^#define MLX_VERSION_(MAJOR|MINOR|PATCH) ([0-9]+)$/\2/p' \
    "$MLXSERVE_SOURCE_DIR/lib/mlx-src/mlx/version.h" | paste -sd. -)"
cmake -S "$(mlx_source "$version")" -B "$MLXSERVE_BUILD_DIR/mlx" -G "Unix Makefiles" \
    -DCMAKE_TOOLCHAIN_FILE="$DEPS_DIR/ios.cmake" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    -DCMAKE_INSTALL_PREFIX="$MLXSERVE_STAGE_DIR" >/dev/null
cmake --build "$MLXSERVE_BUILD_DIR/mlx" --target install >/dev/null
cmake -S "$MLXSERVE_SOURCE_DIR/lib/mlxc-src" -B "$MLXSERVE_BUILD_DIR/mlxc" -G "Unix Makefiles" \
    -DCMAKE_TOOLCHAIN_FILE="$DEPS_DIR/ios.cmake" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DMLX_C_USE_SYSTEM_MLX=ON -DMLX_C_BUILD_EXAMPLES=OFF -DCMAKE_FIND_ROOT_PATH="$MLXSERVE_STAGE_DIR" \
    -DCMAKE_PREFIX_PATH="$MLXSERVE_STAGE_DIR" -DCMAKE_INSTALL_PREFIX="$MLXSERVE_STAGE_DIR" >/dev/null
cmake --build "$MLXSERVE_BUILD_DIR/mlxc" --target install >/dev/null

rsync -a --delete "$MLXSERVE_SOURCE_DIR/.zig-toolchain/lib/" "$MLXSERVE_BUILD_DIR/zig-lib/"
patch -d "$MLXSERVE_BUILD_DIR/zig-lib" -p1 < "$APP_DIR/patches/zig-ios-null-file.patch" >/dev/null
(cd "$MLXSERVE_DIR" && ZIG_LIB_DIR="$MLXSERVE_BUILD_DIR/zig-lib" ZIG_GLOBAL_CACHE_DIR="$MLXSERVE_DIR/.zig-cache/global" \
    "$MLXSERVE_SOURCE_DIR/.zig-toolchain/zig" build -Doptimize=ReleaseFast -Dtarget="aarch64-ios.$IPHONEOS_DEPLOYMENT_TARGET" \
    -Dmlx-dir="$MLXSERVE_STAGE_DIR" --prefix "$MLXSERVE_BUILD_DIR/zig")

framework Mlxserve mlxserve
xcrun --sdk iphoneos clang++ -target "arm64-apple-ios$IPHONEOS_DEPLOYMENT_TARGET" -dynamiclib \
    -o "$FRAMEWORK_DIR/BenchmarksMlxserve" -install_name @rpath/BenchmarksMlxserve.framework/BenchmarksMlxserve \
    -Wl,-force_load,"$MLXSERVE_BUILD_DIR/zig/lib/libengine-mlxserve.a" "$MLXSERVE_STAGE_DIR/lib/libmlxc.a" \
    "$MLXSERVE_STAGE_DIR/lib/libmlx.a" "$MLXSERVE_BUILD_DIR/mlx/mlx/io/libgguflib.a" \
    -framework Accelerate -framework CoreFoundation -framework Foundation -framework IOKit -framework IOSurface \
    -framework Metal -framework QuartzCore -Wl,-exported_symbol,_benchmarks_mlxserve_main
cp "$MLXSERVE_STAGE_DIR/lib/mlx.metallib" "$FRAMEWORK_DIR/"
