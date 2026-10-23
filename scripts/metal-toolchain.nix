pkgs: {
  build,
  hash,
}: let
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
in
  pkgs.runCommand "metal-toolchain-${build}"
  {
    version = build;
    nativeBuildInputs = with pkgs; [aar-extract _7zz curl jq python3 cacert];
    outputHashMode = "recursive";
    outputHashAlgo = "sha256";
    outputHash = hash;
    meta.platforms = ["aarch64-darwin" "x86_64-darwin"];
  }
  ''
    python3 ${./metal-toolchain.py} get ${build} > asset.json

    curl -fL --retry 3 -o toolchain.aar "$(jq -r .url asset.json)"
    echo "$(jq -r .sha256 asset.json)  toolchain.aar" | sha256sum -c -

    mkdir aar dmg
    AEA_KEY="$(jq -r .aea_key asset.json)" aar-extract toolchain.aar aar
    7zz x -y aar/AssetData/Restore/*.dmg -odmg

    cp -r dmg/Metal.xctoolchain/ "$out"/
    ln -s usr/bin "$out"/bin
  ''
