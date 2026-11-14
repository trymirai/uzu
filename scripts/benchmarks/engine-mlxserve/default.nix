{
  pkgs,
  native,
  sources,
  benchmarksPython,
}: let
  source = sources.mlxserve;
  mlx = native {
    name = "mlx-serve-mlx";
    src = source + "/lib/mlx-src";
    nativeBuildInputs = [pkgs.cmake pkgs.ninja pkgs.zsh pkgs.darwin.DarwinTools];
    buildInputs = [pkgs.fmt];
    cmakeFlags = [
      "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
      "-DFETCHCONTENT_SOURCE_DIR_METAL_CPP=${sources.metal-cpp}"
      "-DFETCHCONTENT_SOURCE_DIR_JSON=${sources.json}"
      "-DFETCHCONTENT_SOURCE_DIR_GGUFLIB=${sources.gguf}"
      "-DUSE_SYSTEM_FMT=ON"
      "-DBUILD_SHARED_LIBS=ON"
      "-DMLX_BUILD_TESTS=OFF"
      "-DMLX_BUILD_EXAMPLES=OFF"
      "-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2"
    ];
    postInstall = ''strings "$out/lib/mlx.metallib" | grep _nax > /dev/null'';
  };
  mlxc = native {
    name = "mlx-serve-mlx-c";
    src = source + "/lib/mlxc-src";
    patches = ["${source}/patches/mlxc-gather-qmm-global-scale.patch"];
    nativeBuildInputs = [pkgs.cmake pkgs.ninja];
    buildInputs = [mlx];
    cmakeFlags = ["-DBUILD_SHARED_LIBS=ON" "-DMLX_C_USE_SYSTEM_MLX=ON" "-DMLX_C_BUILD_EXAMPLES=OFF" "-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2"];
  };
in
  native {
    name = "bench-mlxserve";
    src = source;
    patches = pkgs.lib.filesystem.listFilesRecursive ./patches;
    nativeBuildInputs = [pkgs.zig_0_17 pkgs.pkg-config pkgs.makeWrapper];
    buildInputs = [pkgs.libwebp mlxc mlx];
    postPatch = ''
      cp ${./src}/* src/
      cp ${./build.zig} build-benchmarks.zig
    '';
    dontSetZigDefaultFlags = true;
    zigBuildFlags = [
      "--build-file"
      "build-benchmarks.zig"
      "-Doptimize=ReleaseFast"
      "-Dcommon-dir=${../common-cpp/src}"
      "-Dmlx-dir=${mlxc}"
    ];
    postFixup = ''
      makeWrapper ${pkgs.bash}/bin/bash "$out/bin/bench-mlxserve" \
        --add-flags ${./run.sh} \
        --set BENCHMARK_MLXSERVE_BINARY "$out/bin/engine-mlxserve" \
        --set BENCHMARK_MODEL_PYTHON ${benchmarksPython}/bin/python \
        --prefix PATH : ${pkgs.lib.makeBinPath [pkgs.coreutils]}
    '';
  }
