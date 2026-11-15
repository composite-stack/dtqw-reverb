#!/usr/bin/env bash
# AXIOM DTQW — optionale Installation (VST3 + AU in den Benutzerordner, ohne Passwort)
set -euo pipefail
cd "$(dirname "$0")"
NAME="AXIOM DTQW"
install_one() {
  local src="$1" dst="$2"
  [ -d "$src" ] || { echo "Nicht gefunden: $src"; return 0; }
  mkdir -p "$dst"; rm -rf "$dst/$src"
  ditto "$src" "$dst/$src"
  xattr -cr "$dst/$src" 2>/dev/null || true
  codesign --force --deep --timestamp=none --sign - "$dst/$src" >/dev/null 2>&1 || true
  echo "Installiert: $dst/$src"
}
install_one "$NAME.vst3" "$HOME/Library/Audio/Plug-Ins/VST3"
install_one "$NAME.component" "$HOME/Library/Audio/Plug-Ins/Components"
killall -9 AudioComponentRegistrar >/dev/null 2>&1 || true
if [ -d "$NAME.app" ]; then xattr -cr "$NAME.app" 2>/dev/null || true; fi
echo "Fertig. DAW neu starten bzw. Plug-ins neu scannen. Die Standalone-App bleibt in diesem Ordner."
