{
  pkgs,
  native,
  sources,
  server,
}: let
  engine = native {
    name = "tensorfold-native";
    src = sources.tensorfold;
    patches = [./sdk-paths.patch];
    nativeBuildInputs = [pkgs.zig_0_17];
    dontSetZigDefaultFlags = true;
    # WORKAROUND: TensorFold picks the Metal version from the host macOS; pin the one apple-sdk_26 supports.
    zigBuildFlags = ["native" "-Doptimize=ReleaseFast" "-Dmetal-std=metal4.0"];
    installPhase = ''install -Dm755 zig-out/native/bin/tensorfold-native "$out/bin/tensorfold-native"'';
  };
in
  server "tensorfold" {BENCHMARK_TENSORFOLD_BINARY = "${engine}/bin/tensorfold-native";}
