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

        rustToolchainNixCC = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        rustToolchainHostCC = rustToolchainNixCC.overrideAttrs (_: {
          depsHostHostPropagated = [];
          propagatedBuildInputs = [];
          depsTargetTargetPropagated = [];
        });
        craneLibNixCC = (crane.mkLib pkgs).overrideToolchain rustToolchainNixCC;

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

        shader-slang-env = with pkgs; let
          libenv =
            {
              darwin = "DYLD_FALLBACK_LIBRARY_PATH";
              linux = "LD_LIBRARY_PATH";
            }.${
              stdenv.hostPlatform.parsed.kernel.name
            };
        in
          makeSetupHook {
            name = "shader-slang-env";
            propagatedBuildInputs = [shader-slang];
          }
          (writeText "shader-slang-env.sh" ''
            export SLANG_INCLUDE_DIR="${lib.getDev shader-slang}/include"
            export SLANG_LIB_DIR="${lib.getLib shader-slang}/lib"
            addToSearchPath ${libenv} "${lib.getLib shader-slang}/lib"
          '');

        bindgen-env = with pkgs;
          makeSetupHook {name = "bindgen-env";}
          (writeText "bindgen-env.sh" ''
            export LIBCLANG_PATH="${lib.getLib libclang}/lib"
            export BINDGEN_EXTRA_CLANG_ARGS="-isystem ${lib.getDev glibc}/include"
          '');

        vulkan-env = with pkgs;
          makeSetupHook {name = "vulkan-env";}
          (writeText "vulkan-env.sh" ''
            addToSearchPath LD_LIBRARY_PATH "${lib.getLib vulkan-loader}/lib"
            export VK_ADD_LAYER_PATH="${vulkan-validation-layers}/share/vulkan/explicit_layer.d"
          '');

        playwright-env = with pkgs;
          makeSetupHook {
            name = "playwright-env";
            propagatedBuildInputs = [playwright-driver playwright-driver.browsers-chromium];
          }
          (writeText "playwright-env.sh" ''
            export PLAYWRIGHT_BROWSERS_PATH="${playwright-driver.browsers-chromium}"
            export PLAYWRIGHT_DRIVER_PATH="${playwright-driver}"
            export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD="1"
          '');

        hostToolchainBuildInputs = with pkgs; (lib.optionals stdenv.hostPlatform.isLinux [alsa-lib]);
        nixToolchainBuildInputs = hostToolchainBuildInputs ++ (with pkgs; lib.optionals stdenv.hostPlatform.isDarwin [apple-sdk_26]);

        hostToolchainNativeBuildInputs = with pkgs; [cmake] ++ (lib.optionals stdenv.hostPlatform.isLinux [pkg-config]);
        nixToolchainNativeBuildInputs = hostToolchainNativeBuildInputs ++ (with pkgs; lib.optionals stdenv.hostPlatform.isDarwin [metal-toolchain]);

        commonDevshellPackages = with pkgs;
          [
            nil
            shader-slang-env
            playwright-env
            uv
            nodejs
            pnpm_10
            wasm-bindgen-cli_0_2_129 # Keep in sync with Cargo.toml.
            wasmtime
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
          ++ (with pkgs; lib.optionals stdenv.hostPlatform.isLinux [bindgen-env vulkan-env vulkan-tools shaderc spirv-tools lldb])
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
