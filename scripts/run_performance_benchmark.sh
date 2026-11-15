#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# DTQW Reverb - Reproducible Performance Benchmark & Regression Runner
# Kompiliert das Benchmark-Target im Release-Modus (-O3, -DNDEBUG) und vergleicht
# die gemessenen Echtzeit-Metriken mit der Baseline.
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TOLERANCE="25.0"
BASELINE="$ROOT_DIR/performance_baseline.json"
SUMMARY_FILE=""
RUNS="11"
WARMUP="3"
DURATION="1.5"
QUIET=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tolerance)
            TOLERANCE="$2"
            shift 2
            ;;
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --summary)
            SUMMARY_FILE="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        --help|-h)
            echo "Verwendung: $0 [--tolerance <pct>] [--baseline <file>] [--summary <file>] [--runs <n>] [--warmup <n>] [--duration <s>] [--quiet]"
            exit 0
            ;;
        *)
            echo "Unbekannte Option: $1"
            exit 1
            ;;
    esac
done

CXX="${CXX:-g++}"
BUILD_TMP="/tmp/dtqw_bench_run_$$"
mkdir -p "$BUILD_TMP"
trap 'rm -rf "$BUILD_TMP"' EXIT

# Dateien in isoliertes Build-Verzeichnis kopieren
cp "$ROOT_DIR/DTQWReverb.hpp" "$BUILD_TMP/"
cp "$ROOT_DIR/DTQWPerformanceBenchmark.cpp" "$BUILD_TMP/"
if [ -f "$ROOT_DIR/NEON_2_SSE.h" ]; then
    cp "$ROOT_DIR/NEON_2_SSE.h" "$BUILD_TMP/"
fi

cd "$BUILD_TMP"

ARCH_FLAGS=""
if [[ "$(uname -m)" =~ (x86_64|amd64) ]]; then
    ARCH_FLAGS="-msse4.1"
fi

OPT_FLAGS=(
    -std=c++17
    -O3
    -DNDEBUG
    $ARCH_FLAGS
    -Wall
    -Wextra
    -Wno-deprecated-declarations
    -I"$BUILD_TMP"
)

if [ "$QUIET" = false ]; then
    echo "==============================================================================="
    echo "  DTQW DSP PERFORMANCE BENCHMARK & REGRESSION GATE"
    echo "  Compiler: $CXX | Architektur: $(uname -m)"
    echo "  Toleranzlimit: +${TOLERANCE}%"
    echo "==============================================================================="
    echo "--> Kompiliere DTQWPerformanceBenchmark (-O3 -DNDEBUG)..."
fi

$CXX "${OPT_FLAGS[@]}" DTQWPerformanceBenchmark.cpp -o run_benchmark

JSON_OUT="$BUILD_TMP/benchmark_current.json"

if [ "$QUIET" = false ]; then
    echo "--> Führe Benchmark durch (${RUNS} Messungen, ${WARMUP} Warmups, ${DURATION}s Audio)..."
fi

./run_benchmark \
    --runs "$RUNS" \
    --warmup "$WARMUP" \
    --duration "$DURATION" \
    --json "$JSON_OUT" \
    $( [ "$QUIET" = true ] && echo "--quiet" )

# Regression Check
CHECK_ARGS=(
    --current "$JSON_OUT"
    --baseline "$BASELINE"
    --tolerance "$TOLERANCE"
)

if [ -n "$SUMMARY_FILE" ]; then
    CHECK_ARGS+=(--summary "$SUMMARY_FILE")
fi

python3 "$ROOT_DIR/scripts/check_performance_regression.py" "${CHECK_ARGS[@]}"
