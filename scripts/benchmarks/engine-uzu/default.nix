{
  pkgs,
  craneLib,
  metal-toolchain,
}:
craneLib.buildPackage {
  pname = "bench-uzu";
  src = pkgs.lib.fileset.toSource {
    root = ../../..;
    fileset = pkgs.lib.fileset.unions [
      (craneLib.fileset.cargoTomlAndLock ../../..)
      (craneLib.fileset.configToml ../../..)
      ../../../crates
      (craneLib.fileset.commonCargoSources ../../../apps/mirai-gui/desktop)
      (craneLib.fileset.commonCargoSources ./.)
      ../common-cpp/src
    ];
  };
  nativeBuildInputs = [pkgs.cmake pkgs.rustPlatform.bindgenHook metal-toolchain];
  buildInputs = [pkgs.apple-sdk_26];
  cargoExtraArgs = "-p benchmarks-uzu";
  doCheck = false;
  doNotPostBuildInstallCargoBinaries = true;
  installPhaseCommand = ''install -Dm755 target/release/bench-uzu "$out/bin/bench-uzu"'';
  meta.platforms = ["aarch64-darwin"];
}
