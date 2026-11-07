{
  pkgs,
  craneLib,
  metal-toolchain,
}:
craneLib.buildPackage {
  pname = "benchmarks-uzu";
  src = pkgs.lib.cleanSource ../../..;
  nativeBuildInputs = [pkgs.cmake metal-toolchain];
  buildInputs = [pkgs.apple-sdk_26];
  cargoExtraArgs = "-p benchmarks-uzu";
  doCheck = false;
  doNotPostBuildInstallCargoBinaries = true;
  installPhaseCommand = ''install -Dm755 target/release/benchmarks-uzu "$out/bin/benchmarks-uzu"'';
  meta.platforms = ["aarch64-darwin"];
}
