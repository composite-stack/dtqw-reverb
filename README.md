# DTQW Reverb - Discrete-Time Quantum Walk Audio Reverb Engine

Ein modularer, hocheffizienter C++17 Reverb-DSP-Kern basierend auf Discrete-Time Quantum Walks (DTQW) auf einem 4D-Hypercube-Graphen (16 Knoten, 64 gerichtete Verzögerungskanten) mit unitärem Grover-Coin, stufenloser Phasenrotation, Allpass-Dispersion, reziprokem Chorusing und dualem Allpass-Zustandscrossfading.

## Features

- **4D-Hypercube-Topologie:** 16 Knoten, 64 gerichtete Delay-Linien mit SIMD-Transpositionsleser und Guard-Zone.
- **Unitärer Grover-Streu-Coin mit Phasenrotation:** Stufenlose $C(\theta)$-Phasenrotation ohne $t=0$-Wet-Durchschlag.
- **Hadamard-Matrixing:** Orthogonale 16-Knoten-Injektion und -Auskopplung zur Modenentkopplung und Raumabbildung.
- **Frequenzabhängige Allpass-Dispersion:** Kaskadierte Allpass-Filterstufen in den Kanten mit kubischem Smoothstep-State-Crossfading.
- **SIMD-Vektorisierung:** Paritätsgeprüfte Backends für SSE4.1 (x86_64), Apple Silicon ARM NEON und skalare Referenz.
- **Allokationsfreier Audiopfad:** 0 Heap-Allokationen im Audio-Thread nach `prepare()`.

## Build & Tests

### Lokale Verifikation (ASan & UBSan)
```bash
./run_sanitizers.sh
```

### Performance Benchmark & Regression Check
```bash
./scripts/run_performance_benchmark.sh --tolerance 25.0
```

### Manuelle Test-Kompilierung
```bash
g++ -std=c++17 -O3 -msse4.1 -I. DTQWPerformanceBenchmark.cpp -o dtqw_benchmark
./dtqw_benchmark --runs 11 --warmup 3 --duration 1.5 --json benchmark_results.json
```

## CI / CD & Quality Gates

1. **`allpass-crossfade-gate` (Pflicht-Check auf `main`):**
   - Führt Worst-Case-Stresstests (Blockgrößen 1..64, Koeffizienten-Stress, überlappendes Retriggering) unter Clang/GCC AddressSanitizer und UndefinedBehaviorSanitizer auf Linux und macOS aus.
2. **`performance-regression-gate`:**
   - Misst die DSP-Verarbeitungsgeschwindigkeit über standardisierte Szenarien (48 kHz / 96 kHz, Block 64 / 128, Steady-State und dynamischer Crossfade-Stress).
   - Vergleicht die Ergebnisse gegen `performance_baseline.json` und schlägt bei einer Überschreitung des Toleranzlimits (z. B. >25% Verlangsamung) fehl.
