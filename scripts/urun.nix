{
  writeShellApplication,
  coreutils,
  direnv,
  git,
  jujutsu,
  openssh,
  rsync,
}:
writeShellApplication {
  name = "urun";
  runtimeInputs = [coreutils direnv git jujutsu openssh rsync];
  text = builtins.readFile ./urun;
}
