# DTQW Reverb DSP Core

Framework-independent C++17 implementation of a 4D-hypercube discrete-time quantum-walk reverb engine.

## Offline WAV render harness

`dtqw_render_wav` lets you audition the DSP without a DAW/plugin wrapper. It accepts RIFF/WAVE PCM 16/24/32-bit integer and IEEE float 32-bit input, converts mono to dual-mono, and writes stereo PCM 16/24-bit or float32 WAV. The input sample rate is preserved; no resampling is performed.

Build with CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDTQW_ENABLE_SANITIZERS=OFF
cmake --build build --target dtqw_render_wav dtqw_wav_harness_tests
ctest --test-dir build -R WavHarnessTests --output-on-failure
```

### Render an input file

```bash
./build/dtqw_render_wav input.wav output.wav \
  --room-size 1.2 --decay 2.5 --damping 5500 --mix 0.25 \
  --coin-phase-deg 45 --dispersion 0.5 --chorus-rate 0.8 --chorus-depth 3
```

### Generate a stereo impulse response

```bash
./build/dtqw_render_wav --ir hall_ir.wav --duration 4 --sample-rate 48000 \
  --float --decay 2.8 --dispersion 0.4
```

Run `./build/dtqw_render_wav --help` for the full options list. `--normalize` buffers the rendered output in memory to find its peak; omit it for streaming WAV output with fixed gain. Defaults are 100% wet for an IR and 35% wet for file processing, with a tail of up to 8 seconds added after the input.

## DSP validation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The project includes SIMD parity, automation, dispersion/crossfade, stress and performance regression tests. `dtqw_render_wav` is an offline listening/testing utility and does not replace a real-time plugin wrapper.
