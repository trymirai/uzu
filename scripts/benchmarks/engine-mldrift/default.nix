{
  pkgs,
  craneLib,
  sources,
}: let
  inherit (pkgs) lib;
  rustTools = pkgs.rust-bin.stable.latest.minimal;
  cargoBazel = (craneLib.overrideToolchain rustTools).buildPackage {
    src = sources.rules-rust + "/crate_universe";
    buildInputs = [pkgs.libz];
    doCheck = false;
    postPatch = "patchShebangs --build src/metadata/cargo_tree_rustc_wrapper.sh";
  };
  rules-cc = pkgs.applyPatches {
    src = sources.rules-cc;
    patches = [./patches/nix-rules-cc-objc.patch];
    nativeBuildInputs = [pkgs.python3];
    postPatch = "patchShebangs cc/private/toolchain";
  };
  environment = pkgs.makeSetupHook {name = "benchmark-bazel";} (pkgs.writeText "benchmark-bazel.sh" ''
    export TEST_TMPDIR="$NIX_BUILD_TOP/bazel"
    mkdir -p "$TEST_TMPDIR"
    export BAZEL_SH=${pkgs.bash}/bin/bash
    benchmarkBazelSetup() {
      echo "build --jobs=$NIX_BUILD_CORES" >> .bazelrc
      if [[ "$command" == build ]]; then
        sed 's|@cargo-bazel@|${cargoBazel}|g' vendor_dir/MODULE.bazel.lock > MODULE.bazel.lock
        substitute vendor_dir/rules_python+/python/private/hermetic_runtime_repo_setup.bzl \
          vendor_dir/rules_python+/python/private/hermetic_runtime_repo_setup.bzl.patched \
          --replace-fail 'py_runtime(' 'py_runtime(stub_shebang = "#!${pkgs.python3}/bin/python3",'
        mv vendor_dir/rules_python+/python/private/hermetic_runtime_repo_setup.bzl{.patched,}
      fi
    }
    preBuildHooks+=(benchmarkBazelSetup)
    benchmarkBazelCleanup() {
      if [[ "$command" == vendor ]]; then
        rm -f vendor_dir/bazel-external
        chmod -R u+w vendor_dir
        find vendor_dir -type d -name __pycache__ -prune -exec rm -rf {} +
        sed 's|${cargoBazel}|@cargo-bazel@|g' MODULE.bazel.lock > vendor_dir/MODULE.bazel.lock
      fi
    }
    postBuildHooks+=(benchmarkBazelCleanup)
  '');
  source = pkgs.applyPatches {
    src = sources.mldrift;
    patches = [./patches/benchmark-metrics.patch];
    postPatch = ''
      cp ${./requirements.lock} requirements.txt
      substituteInPlace third_party/repo.bzl \
        --replace-fail 'ctx.symlink(Label(ctx.attr.build_file), buildfile_path)' \
          'ctx.file(buildfile_path, ctx.read(Label(ctx.attr.build_file)), executable = False)' \
        --replace-fail 'ctx.symlink(Label(internal_src), ctx.path(external_dest))' \
          'ctx.file(external_dest, ctx.read(Label(internal_src)), executable = False)'
      substituteInPlace MODULE.bazel \
        --replace-fail 'pip.parse(' 'pip.parse(download_only = True,' \
        --replace-fail 'crate.from_specs(' 'crate.from_specs(host_tools = "//benchmark-rust:BUILD.bazel",'
      mkdir benchmark-rust
      touch benchmark-rust/BUILD.bazel
      ln -s ${rustTools}/bin benchmark-rust/bin
      mkdir benchmarks
      cp ${./src}/* ${../common-cpp/src}/* benchmarks/
      echo 'local_path_override(module_name = "rules_cc", path = "${rules-cc}")' >> MODULE.bazel
    '';
  };
  bazelPackage = pkgs.callPackage "${pkgs.path}/pkgs/by-name/ba/bazel_8/build-support/bazelPackage.nix" {
    stdenv = pkgs.addAttrsToDerivation {dontCheckForBrokenSymlinks = true;} pkgs.stdenv;
  };
  engine = bazelPackage {
    name = "bench-mldrift";
    src = source;
    bazel = pkgs.bazel_8;
    registry = sources.bazel-registry;
    targets = ["//benchmarks:engine-mldrift" "//ml_drift/samples/llm:extract_weights_hf"];
    nativeBuildInputs = [pkgs.python3 pkgs.git pkgs.darwin.cctools environment];
    buildInputs = [pkgs.apple-sdk_26];
    env = {
      USE_BAZEL_VERSION = pkgs.bazel_8.version;
      BAZEL_USE_CPP_ONLY_TOOLCHAIN = "1";
      SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
      CARGO_HTTP_CAINFO = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
      CARGO_BAZEL_GENERATOR_URL = "file://${cargoBazel}/bin/cargo-bazel";
    };
    commandArgs =
      ["-c" "opt" "--spawn_strategy=processwrapper-sandbox" "--features=-layering_check" "--host_features=-layering_check"]
      ++ lib.concatMap (name: ["--action_env=${name}" "--host_action_env=${name}"]) [
        "SDKROOT"
        "DEVELOPER_DIR"
        "NIX_CFLAGS_COMPILE"
        "NIX_LDFLAGS"
        "NIX_CC_WRAPPER_TARGET_HOST_${pkgs.stdenv.cc.suffixSalt}"
        "NIX_BINTOOLS_WRAPPER_TARGET_HOST_${pkgs.stdenv.cc.bintools.suffixSalt}"
      ];
    bazelVendorDepsFOD = {
      outputHash = "sha256-XAzVXiaR3/nRHrSJDyW+o9jJWS6jI8PkAlQyl7UmRTc=";
      outputHashAlgo = "sha256";
    };
    installPhase = ''
      mkdir -p "$out/bin"
      cp bazel-bin/benchmarks/engine-mldrift bazel-bin/ml_drift/samples/llm/extract_weights_hf "$out/bin/"
      cp -rL bazel-bin/ml_drift/samples/llm/extract_weights_hf.runfiles "$out/bin/"
      rm "$out/bin/extract_weights_hf.runfiles/MANIFEST"
    '';
  };
in
  engine.overrideAttrs (old: {
    nativeBuildInputs = old.nativeBuildInputs ++ [pkgs.makeWrapper];
    meta.platforms = ["aarch64-darwin"];
    stripDebugList = ["bin/engine-mldrift"];
    postFixup = ''
      makeWrapper ${pkgs.bash}/bin/bash "$out/bin/bench-mldrift" \
        --add-flags ${./run.sh} \
        --set BENCHMARK_MLDRIFT_BINARY "$out/bin/engine-mldrift" \
        --set BENCHMARK_MLDRIFT_EXTRACTOR "$out/bin/extract_weights_hf" \
        --set BENCHMARK_MLDRIFT_REVISION ${sources.mldrift.rev} \
        --prefix PATH : ${lib.makeBinPath [pkgs.python3 pkgs.perl pkgs.coreutils pkgs.huggingface-cli]}
    '';
  })
