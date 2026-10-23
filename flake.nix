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
    flake-utils.lib.eachSystem ["aarch64-darwin" "aarch64-linux" "x86_64-linux"] (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [(import rust-overlay)];
        };

        rustToolchainNixCC = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        rustToolchainHostCC = rustToolchainNixCC.overrideAttrs (_: {
          depsHostHostPropagated = [];
          propagatedBuildInputs = [];
          depsTargetTargetPropagated = [];
        });
        craneLibNixCC = (crane.mkLib pkgs).overrideToolchain rustToolchainNixCC;

        metal-toolchain = import ./scripts/metal-toolchain.nix pkgs {
          build = "27A266a";
          hash = "sha256-vpSn2ZVUo5tNRgJ89KMBCmPE4JFefP70DfCarVagniU=";
        };

        hostToolchainBuildInputs = with pkgs; (lib.optionals stdenv.hostPlatform.isLinux [alsa-lib]);
        nixToolchainBuildInputs = hostToolchainBuildInputs ++ (with pkgs; lib.optionals stdenv.hostPlatform.isDarwin [apple-sdk_26]);

        hostToolchainNativeBuildInputs = with pkgs; [cmake] ++ (lib.optionals stdenv.hostPlatform.isLinux [pkg-config]);
        nixToolchainNativeBuildInputs = hostToolchainNativeBuildInputs ++ (with pkgs; lib.optionals stdenv.hostPlatform.isDarwin [metal-toolchain]);

        commonDevshellPackages = with pkgs;
          [
            nil
            uv
            nodejs_26
            pnpm_12
            wasmtime_36
            evcxr
            cargo-deny
            cargo-nextest
            cargo-hack
            cargo-expand
            samply
            cargo-flamegraph
            cargo-show-asm
            critcmp
          ]
          ++ (with pkgs; lib.optionals stdenv.hostPlatform.isDarwin [cargo-swift tuist]);
        hostToolchainDevshellPackages = commonDevshellPackages ++ [rustToolchainHostCC];
        nixToolchainDevshellPackages = commonDevshellPackages ++ [rustToolchainNixCC];

        mirai = craneLibNixCC.buildPackage {
          pname = "mirai";
          src = ./.;

          cargoExtraArgs = "-p cli";
          installPhaseCommand = ''
            mkdir -p $out/bin
            install -Dm755 target/release/cli $out/bin/mirai
          '';

          buildInputs = nixToolchainBuildInputs;
          nativeBuildInputs = nixToolchainNativeBuildInputs;

          doCheck = false;
        };
      in {
        formatter = pkgs.alejandra;

        packages = {
          inherit mirai;
          default = mirai;
        };

        devShells = {
          default = pkgs.mkShell {
            buildInputs = nixToolchainBuildInputs;
            nativeBuildInputs = nixToolchainNativeBuildInputs;

            packages = nixToolchainDevshellPackages;
          };

          host-xcode = pkgs.mkShellNoCC {
            buildInputs = hostToolchainBuildInputs;
            nativeBuildInputs = hostToolchainNativeBuildInputs;

            packages = hostToolchainDevshellPackages;

            TUIST_GENERATE_OPEN = "false";
          };
        };
      }
    );
}
