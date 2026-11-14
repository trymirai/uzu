{
  pkgs,
  craneLib,
  metal-toolchain,
  mirai-cli,
  pyproject-nix,
  uv2nix,
  pyproject-build-systems,
}: let
  inherit (pkgs) lib;
  sources = lib.mapAttrs (_: pin:
    if pin ? url
    then pkgs.fetchzip pin
    else pkgs.fetchFromGitHub pin) (lib.importJSON ./sources.json);
  python = import ./common-py/python.nix {inherit pkgs pyproject-nix uv2nix pyproject-build-systems;};
  benchmarksPython = python {
    projectRoot = ./.;
    name = "benchmarks-env";
  };
  server = name: engine: environment:
    pkgs.runCommand name {
      nativeBuildInputs = [pkgs.makeWrapper];
      meta.platforms = ["aarch64-darwin"];
    } ''
      makeWrapper ${benchmarksPython}/bin/bench-server "$out/bin/${name}" \
        --add-flags ${engine} \
        ${lib.concatMapAttrsStringSep " " (name: value: "--set ${name} ${lib.escapeShellArg value}") environment}
    '';
  native = args:
    pkgs.stdenv.mkDerivation (args
      // {
        nativeBuildInputs = [metal-toolchain pkgs.writableTmpDirAsHomeHook] ++ (args.nativeBuildInputs or []);
        buildInputs = [pkgs.apple-sdk_26] ++ (args.buildInputs or []);
        meta.platforms = ["aarch64-darwin"];
      });
  callPackage = lib.callPackageWith (pkgs // {inherit craneLib metal-toolchain sources python benchmarksPython server native;});
in rec {
  bench-uzu = callPackage ./engine-uzu {};
  bench-uzu-server = server "bench-uzu-server" "uzu" {BENCHMARK_UZU_BINARY = "${mirai-cli}/bin/mirai-cli";};
  bench-splash = callPackage ./engine-splash {};
  bench-splash-server = bench-splash.server;
  bench-mlx = callPackage ./engine-mlx {};
  bench-mtplx = callPackage ./engine-mtplx {};
  bench-omlx = callPackage ./engine-omlx {};
  bench-llamacpp = callPackage ./engine-llamacpp {};
  bench-mlxserve = callPackage ./engine-mlxserve {};
  bench-mldrift = callPackage ./engine-mldrift {};
  bench-magnitude = callPackage ./engine-magnitude {};
  bench-tensorfold = callPackage ./engine-tensorfold {};
}
