{
  pkgs,
  craneLib,
  sources,
  server,
}: let
  source = sources.magnitude;
  toolchain = pkgs.rust-bin.fromRustupToolchainFile "${source}/inference/rust-toolchain.toml";
  engine = (craneLib.overrideToolchain toolchain).buildPackage {
    pname = "magnitude-engine";
    src = source + "/inference";
    cargoExtraArgs = "-p magnitude-engine-cli --bin magnitude-engine";
    nativeBuildInputs = [pkgs.cmake];
    buildInputs = [pkgs.apple-sdk_26];
    doCheck = false;
  };
in
  server "magnitude" {BENCHMARK_MAGNITUDE_BINARY = "${engine}/bin/magnitude-engine";}
