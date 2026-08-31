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
        pkgs = import nixpkgs {
          inherit system;
          overlays = [(import rust-overlay)];
        };

        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        aar-extract = pkgs.ipsw.overrideAttrs (old: {
          pname = "aar-extract";
          postPatch =
            (old.postPatch or "")
            + ''
              mkdir -p cmd/aar-extract
              cat > cmd/aar-extract/main.go <<'EOF'
              package main

              import "io"
              import "os"
              import "path/filepath"
              import "github.com/blacktop/ipsw/pkg/ota"

              func main() {
              	o, err := ota.Open(os.Args[1], &ota.Config{SymmetricKey: os.Getenv("AEA_KEY")})
              	if err != nil {
              		panic(err)
              	}
              	for _, f := range o.Files() {
              		if f.IsDir() {
              			continue
              		}
              		dst := filepath.Join(os.Args[2], f.Name())
              		rc, err := f.Open(false)
              		if err != nil {
              			panic(err)
              		}
              		os.MkdirAll(filepath.Dir(dst), 0o755)
              		out, _ := os.Create(dst)
              		io.Copy(out, rc)
              		rc.Close()
              		out.Close()
              	}
              }
              EOF
            '';
          subPackages = ["cmd/aar-extract"];
          ldflags = [];
        });

        metal-toolchain = pkgs.stdenvNoCC.mkDerivation {
          pname = "metal-toolchain";
          version = "17F109";

          src = pkgs.fetchurl {
            url = "https://updates.cdn-apple.com/2026MobileAssets/mobileassets/022-22264/722AB304-567A-4611-A7CA-1A94FA951B0B/com_apple_MobileAsset_MetalToolchain/78ED9FD0-8B39-4EEF-B0EE-67013D77BE16.aar";
            hash = "sha256-UraVdUkU0VDWMP4UnKzHam1yBS1qYtqx8rkvPmgmelY=";
          };
          dontUnpack = true;

          nativeBuildInputs = with pkgs; [aar-extract _7zz];

          buildPhase = ''
            mkdir aar dmg
            AEA_KEY='iOd2WvA6GXSJJrdQkp/vj1nP6FyjxkWe9JZwvJyG71E=' aar-extract "$src" aar
            7zz x -y aar/AssetData/Restore/022-21788-058.dmg -odmg
          '';

          installPhase = ''
            cp -r dmg/Metal.xctoolchain/ "$out"/
            ln -s usr/bin "$out"/bin
          '';

          meta = {
            platforms = ["aarch64-darwin" "x86_64-darwin"];
          };
        };

        buildInputs = with pkgs; (lib.optionals pkgs.stdenv.hostPlatform.isLinux [pkg-config alsa-lib]) ++ (lib.optionals pkgs.stdenv.hostPlatform.isDarwin [apple-sdk_26]);
        nativeBuildInputs = with pkgs; [cmake] ++ (lib.optionals pkgs.stdenv.hostPlatform.isDarwin [metal-toolchain]);

        mirai = craneLib.buildPackage {
          pname = "mirai";
          src = ./.;

          cargoExtraArgs = "-p cli";
          installPhaseCommand = ''
            mkdir -p $out/bin
            install -Dm755 target/release/cli $out/bin/mirai
          '';

          nativeBuildInputs = nativeBuildInputs ++ (with pkgs; (lib.optionals pkgs.stdenv.hostPlatform.isDarwin [writableTmpDirAsHomeHook]));
          inherit buildInputs;

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
                nil
                uv
                wasmtime
                evcxr
                rustToolchain
                cargo-deny
                cargo-nextest
                cargo-hack
                cargo-expand
                cargo-flamegraph
                cargo-show-asm
                critcmp
              ]);
            inherit buildInputs;
          };
      }
    );
}
