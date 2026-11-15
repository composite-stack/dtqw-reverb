#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# DTQW Reverb - Mandatory CI & Pre-Push Verification Suite
# Prüft Worst-Case-Stresstests, State-Crossfading und Dispersion mit ASan & UBSan.
# -----------------------------------------------------------------------------

FORCE_RUN=false
CHANGED_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --force|--ci)
            FORCE_RUN=true
            ;;
        --changed-only)
            CHANGED_ONLY=true
            ;;
        --help|-h)
            echo "Verwendung: $0 [--force|--ci] [--changed-only]"
            exit 0
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prüfe, ob Änderungen an der Allpass- oder Crossfade-Logik vorliegen
should_run() {
    if [ "$FORCE_RUN" = true ]; then
        return 0
    fi

    # Wenn kein Git vorhanden ist, führe die Tests sicherheitshalber aus
    if ! git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        return 0
    fi

    local DIFF_TARGET="HEAD"
    if git -C "$ROOT_DIR" rev-parse --verify origin/main >/dev/null 2>&1; then
        DIFF_TARGET="origin/main...HEAD"
    elif git -C "$ROOT_DIR" rev-parse --verify HEAD~1 >/dev/null 2>&1; then
        DIFF_TARGET="HEAD~1"
    fi

    local CHANGED_FILES
    CHANGED_FILES=$(git -C "$ROOT_DIR" diff --name-only "$DIFF_TARGET" 2>/dev/null || git -C "$ROOT_DIR" status --porcelain | awk '{print $2}')

    if echo "$CHANGED_FILES" | grep -E "(DTQWReverb\.hpp|DTQWWorstCaseStressTests\.cpp|DTQWStateCrossfadeTests\.cpp|DTQWDispersionTests\.cpp|NEON_2_SSE\.h|scripts/)" >/dev/null 2>&1; then
        echo "==> Relevante Änderungen an Allpass/Crossfade/Stresstests detektiert:"
        echo "$CHANGED_FILES" | grep -E "(DTQWReverb\.hpp|DTQWWorstCaseStressTests\.cpp|DTQWStateCrossfadeTests\.cpp|DTQWDispersionTests\.cpp|NEON_2_SSE\.h|scripts/)" | sed 's/^/    - /'
        return 0
    fi

    if [ "$CHANGED_ONLY" = true ]; then
        echo "==> Keine Änderungen an Allpass- oder Crossfade-Dateien erkannt. CI-Schritt wird übersprungen."
        return 1
    fi

    return 0
}

if ! should_run; then
    exit 0
fi

CXX="${CXX:-g++}"
BUILD_TMP="/tmp/dtqw_ci_allpass_crossfade_$$"
mkdir -p "$BUILD_TMP"
trap 'rm -rf "$BUILD_TMP"' EXIT

echo "==============================================================================="
echo "  DTQW CI GATE: ASan + UBSan Worst-Case-Stresstests & Allpass-Validierung"
echo "  Compiler: $CXX"
echo "  Architektur: $(uname -m), Plattform: $(uname -s)"
echo "==============================================================================="

# Quell- und Testdateien spiegeln
cp "$ROOT_DIR/DTQWReverb.hpp" "$BUILD_TMP/"
cp "$ROOT_DIR/DTQWWorstCaseStressTests.cpp" "$BUILD_TMP/"
cp "$ROOT_DIR/DTQWStateCrossfadeTests.cpp" "$BUILD_TMP/"
cp "$ROOT_DIR/DTQWDispersionTests.cpp" "$BUILD_TMP/"
if [ -f "$ROOT_DIR/NEON_2_SSE.h" ]; then
    cp "$ROOT_DIR/NEON_2_SSE.h" "$BUILD_TMP/"
fi

cd "$BUILD_TMP"

ARCH_FLAGS=""
if [[ "$(uname -m)" =~ (x86_64|amd64) ]]; then
    ARCH_FLAGS="-msse4.1"
fi

COMMON_SAN_FLAGS=(
    -std=c++17
    -O1
    -g
    $ARCH_FLAGS
    -fsanitize=address,undefined
    -fno-omit-frame-pointer
    -Wall
    -Wextra
    -Wno-deprecated-declarations
    -I"$BUILD_TMP"
)

export ASAN_OPTIONS="detect_leaks=1:check_initialization_order=1:strict_init_order=1:detect_stack_use_after_return=1"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"

# 1. Worst-Case-Stresstests (Blockgrößen 1..64, Jittering, dauerhafte Koeffizienten-Automation, schnelles Retriggering)
echo ""
echo "--- [1/3] Kompiliere & führe DTQWWorstCaseStressTests aus (ASan + UBSan) ---"
$CXX "${COMMON_SAN_FLAGS[@]}" DTQWWorstCaseStressTests.cpp -o run_worstcase_stress
./run_worstcase_stress
echo ">>> [1/3] DTQWWorstCaseStressTests: ERFOLGREICH BESTANDEN."

# 2. State-Crossfade-Tests (Zustandsvererbung, Klickunterdrückung, überlappende Blends)
echo ""
echo "--- [2/3] Kompiliere & führe DTQWStateCrossfadeTests aus (ASan + UBSan) ---"
$CXX "${COMMON_SAN_FLAGS[@]}" DTQWStateCrossfadeTests.cpp -o run_state_crossfade
./run_state_crossfade
echo ">>> [2/3] DTQWStateCrossfadeTests: ERFOLGREICH BESTANDEN."

# 3. Dispersions- und Frequenzgangtests (Unitäre Stabilität, Gruppenlaufzeitkurven)
echo ""
echo "--- [3/3] Kompiliere & führe DTQWDispersionTests aus (ASan + UBSan) ---"
$CXX "${COMMON_SAN_FLAGS[@]}" DTQWDispersionTests.cpp -o run_dispersion
./run_dispersion
echo ">>> [3/3] DTQWDispersionTests: ERFOLGREICH BESTANDEN."

echo ""
echo "==============================================================================="
echo "  ✅ CI GATE PASSED: Alle Allpass-, Crossfade- und Worst-Case-Tests fehlerfrei!"
echo "  0 Memory Leaks, 0 Buffer-Overflows, 0 Undefined Behavior, 0 Heap-Allokationen."
echo "==============================================================================="
