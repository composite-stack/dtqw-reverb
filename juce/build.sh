#!/usr/bin/env bash
# AXIOM DTQW — JUCE Build (macOS, ein Befehl)
#   bash build.sh
# Ergebnis: ~/Downloads/AXIOM-DTQW-Plugins_vN/
#   AXIOM DTQW.vst3        VST3  (Cubase, Ableton, Reaper, Studio One, Bitwig …)
#   AXIOM DTQW.component   Audio Unit (Logic, GarageBand)
#   AXIOM DTQW.app         Standalone-Programm
#   install-mac.command    optional: installiert VST3/AU in deinen Benutzerordner
#   AXIOM-DTQW-vN-mac.zip  versandfertiges Paket
# Universal Binary (Apple Silicon + Intel), macOS 10.13+.
# Es wird NICHTS installiert — die Dateien landen nur im Download-Ordner.
set -Eeuo pipefail
cd "$(dirname "$0")"

VERSION="$(tr -d '[:space:]' < VERSION 2>/dev/null || echo 1)"
NAME="AXIOM DTQW"
REL="build/AxiomDTQW_artefacts/Release"
OUT="$HOME/Downloads/AXIOM-DTQW-Plugins_v$VERSION"
CACHE="$HOME/.cache/AXIOM-DTQW"
JUCE_DIR="$CACHE/JUCE-8.0.12"

on_error() {
  echo
  echo "!!! Abbruch bei Schritt: ${STEP:-?}"
  echo "!!! Bitte die letzten ~20 Zeilen oberhalb (meist mit 'error:') kopieren und schicken."
}
trap on_error ERR

[ "$(uname -s)" = "Darwin" ] || { echo "Dieses Skript ist für macOS gedacht."; exit 1; }

STEP="1/6 Werkzeuge"
echo "==> [1/6] Werkzeuge prüfen (AXIOM DTQW v$VERSION)"
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Apple-Buildwerkzeuge fehlen — das Installationsfenster wird geöffnet."
  echo "Nach der Installation denselben Terminal-Befehl einfach erneut starten."
  xcode-select --install || true
  trap - ERR; exit 0
fi
if ! command -v cmake >/dev/null 2>&1; then
  for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do if [ -x "$b" ]; then eval "$("$b" shellenv)"; fi; done
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew wird installiert (macOS fragt einmalig nach deinem Passwort)."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do if [ -x "$b" ]; then eval "$("$b" shellenv)"; fi; done
  fi
  brew install cmake
fi
echo "   cmake $(cmake --version | head -1 | awk '{print $3}') · $(uname -m) Mac"

STEP="2/6 JUCE"
echo "==> [2/6] JUCE 8.0.12 bereitstellen (einmalig, danach aus dem Cache)"
mkdir -p "$CACHE"
if [ ! -f "$JUCE_DIR/CMakeLists.txt" ]; then
  rm -rf "$JUCE_DIR"
  git clone --depth 1 --branch 8.0.12 https://github.com/juce-framework/JUCE.git "$JUCE_DIR"
fi

STEP="3/6 Konfigurieren"
echo "==> [3/6] Konfigurieren (Release, Universal arm64 + x86_64)"
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" -DCMAKE_OSX_DEPLOYMENT_TARGET=10.13 \
  -DFETCHCONTENT_SOURCE_DIR_JUCE="$JUCE_DIR"

STEP="4/6 Bauen"
echo "==> [4/6] Bauen (VST3 + AU + Standalone) — beim ersten Mal einige Minuten"
cmake --build build --config Release -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"

STEP="5/6 DSP-Test"
echo "==> [5/6] DSP-Selbsttest"
c++ -std=c++17 -O2 -I Source/Dsp -o /tmp/axiom_dtqw_parity tests/parity_main.cpp
/tmp/axiom_dtqw_parity > /tmp/axiom_dtqw_selftest.txt
cat /tmp/axiom_dtqw_selftest.txt
if grep -qiE "nan|inf|fail|error" /tmp/axiom_dtqw_selftest.txt; then
  echo "   DSP-Selbsttest: FEHLER"; exit 1;
fi
echo "   DSP-Selbsttest: OK"
rm -f /tmp/axiom_dtqw_parity /tmp/axiom_dtqw_selftest.txt

STEP="6/6 Ergebnisordner"
echo "==> [6/6] Signieren (ad-hoc) & Ergebnisordner in Downloads anlegen"
for b in "$REL/VST3/$NAME.vst3" "$REL/AU/$NAME.component" "$REL/Standalone/$NAME.app"; do
  [ -e "$b" ] || { echo "   Fehlt: $b — Bau unvollständig."; exit 1; }
  xattr -cr "$b" 2>/dev/null || true
  codesign --force --deep --timestamp=none --sign - "$b"
  codesign --verify --deep --strict "$b"
done
ARCHS="$(lipo -archs "$REL/VST3/$NAME.vst3/Contents/MacOS/$NAME" 2>/dev/null || echo '?')"
case "$ARCHS" in *x86_64*arm64*|*arm64*x86_64*) ;; *) echo "   Warnung: kein Universal Binary ($ARCHS)";; esac
# Nur EIN Ergebnisordner in Downloads: ältere AXIOM-DTQW-Ordner entfernen
for old in "$HOME/Downloads"/AXIOM-DTQW-Plugins_v*; do
  if [ -d "$old" ] && [ "$old" != "$OUT" ]; then rm -rf "$old"; fi
done
rm -rf "$OUT"; mkdir -p "$OUT"
ditto "$REL/VST3/$NAME.vst3" "$OUT/$NAME.vst3"
ditto "$REL/AU/$NAME.component" "$OUT/$NAME.component"
ditto "$REL/Standalone/$NAME.app" "$OUT/$NAME.app"
cp install-mac.command "$OUT/" && chmod +x "$OUT/install-mac.command"
cat > "$OUT/LIES MICH.txt" <<TXT
AXIOM DTQW v$VERSION — fertige Plug-ins (Universal: Apple Silicon + Intel, macOS 10.13+)
Architekturen: $ARCHS

Schnellstart:
  1. Standalone-App: "AXIOM DTQW.app" per Doppelklick öffnen — spielt sofort mit Mikrofon/Eingang.
  2. DAW: "install-mac.command" doppelklicken kopiert VST3 und AU nach:
       ~/Library/Audio/Plug-Ins/VST3/
       ~/Library/Audio/Plug-Ins/Components/
     Danach DAW starten oder Plug-in-Scan ausführen.

Deinstallation:
  Ordner ~/Downloads/AXIOM-DTQW-Plugins_v$VERSION löschen,
  und die Dateien aus den beiden Audio-Plug-Ins-Ordnern entfernen.
TXT

# Fertiges Zip-Paket für Weitergabe erzeugen
(cd "$OUT" && zip -q -r -X "$OUT/AXIOM-DTQW-v$VERSION-mac.zip" "$NAME.vst3" "$NAME.component" "$NAME.app" "install-mac.command" "LIES MICH.txt")

echo
echo "================================================================================"
echo "FERTIG: AXIOM DTQW v$VERSION liegt in deinem Download-Ordner:"
echo "   $OUT"
echo "   • $NAME.vst3"
echo "   • $NAME.component"
echo "   • $NAME.app"
echo "   • install-mac.command (Doppelklick = Installation ohne Passwort)"
echo "   • AXIOM-DTQW-v$VERSION-mac.zip"
echo "================================================================================"
open "$OUT" 2>/dev/null || true
