{
  description = "A high-performance inference engine for AI models";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    nixpkgs,
    rust-overlay,
    crane,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlays = [(import rust-overlay)];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        nativeBuildInputs = with pkgs;
          [
            # for xgrammar-rs
            cmake
          ]
          ++ (pkgs.lib.optionals pkgs.stdenv.isDarwin [
            # impure hack to get metal toolchain
            (lib.hiPrio (writeShellScriptBin "xcrun" ''
              unset DEVELOPER_DIR
              exec /usr/bin/xcrun "$@"
            ''))
          ]);

        mirai = craneLib.buildPackage {
          pname = "mirai";
          src = ./.;

          cargoExtraArgs = "-p cli";
          installPhaseCommand = ''
            mkdir -p $out/bin
            install -Dm755 target/release/cli $out/bin/mirai
          '';

          inherit nativeBuildInputs;

          doCheck = false;
        };
      in {
        formatter = pkgs.alejandra;

        packages = {
          inherit mirai;
          default = mirai;
        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs =
            nativeBuildInputs
            ++ (with pkgs; [
              nixd
              uv
              wasmtime
              evcxr
              rustToolchain
              cargo-deny
              cargo-nextest
              cargo-flamegraph
              cargo-show-asm
              critcmp
            ]);
        };
      }
    );
}
