#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-render-wav}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DDTQW_ENABLE_SANITIZERS=OFF
cmake --build "$BUILD_DIR" --target dtqw_render_wav dtqw_wav_harness_tests --parallel
ctest --test-dir "$BUILD_DIR" -R WavHarnessTests --output-on-failure

echo "Ready: $BUILD_DIR/dtqw_render_wav"
