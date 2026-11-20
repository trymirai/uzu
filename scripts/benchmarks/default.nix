{
  pkgs,
  craneLib,
  metal-toolchain,
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
      makeWrapper ${benchmarksPython}/bin/server "$out/bin/${name}" \
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
in {
  engine-uzu = callPackage ./engine-uzu {};
  engine-splash = callPackage ./engine-splash {};
  engine-mlx = callPackage ./engine-mlx {};
  engine-mtplx = callPackage ./engine-mtplx {};
  engine-omlx = callPackage ./engine-omlx {};
  engine-llamacpp = callPackage ./engine-llamacpp {};
  engine-mlxserve = callPackage ./engine-mlxserve {};
  engine-magnitude = callPackage ./engine-magnitude {};
  engine-tensorfold = callPackage ./engine-tensorfold {};
  engine-kapitsa = callPackage ./engine-kapitsa {};
}
