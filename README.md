# DTQW Reverb Engine

Ein modulares C++17 Digital Signal Processing (DSP) Reverb-System, basierend auf einem **Discrete-Time Quantum Walk (DTQW)** auf einem **4D-Hypercube-Graphen**. 

Das Netzwerk interferiert Delays konstruktiv und destruktiv über orthogonale Kanten-Coins und erzeugt ballistische Ausbreitungsmuster sowie eine resonante Modendichte, die mit klassischen Markov-basierten Feedback Delay Networks (FDNs) oder Waveguide-Meshes nicht erreichbar ist.

---

## 🏛️ Mathematische Architektur

* **Topologie:** 4D-Hypercube mit $V = 16$ Knoten ($u \in \{0, \dots, 15\}$), regulärem Grad $d = 4$ und $E = 64$ gerichteten Kanten/Delay-Ports.
* **Kanten-Permutation (Shift-Operator):**
  $$S(u, k) = (u \oplus 2^k, k) \quad \text{mit } S^2 = I$$
* **Knoten-Streuung (Unitärer Grover-Coin):**
  $$C = \frac{1}{2} \mathbf{1}\mathbf{1}^T - I$$
* **Stufenlose komplexe Coin-Phasendrehung:**
  Erhält Injektions- ($e_0$) und Auslesemoden ($e_1$), während der orthogonale Transversalraum $(e_2, e_3)$ mit Rotationswinkel $\theta$ rotiert. Dies erlaubt eine kontinuierliche Klangfarben- und Diffusionssteuerung ohne Energieverlust.
* **Frequenzabhängige Dispersion:**
  SIMD-beschleunigte Allpass-Kaskaden in den Kantenleitungen zur Simulation von Phasendispersion (Chirped Wavepackets) mit stufenloser Gruppenlaufzeit-Einstellung.
* **Duales Allpass-State-Crossfading:**
  Klickfreie Host-Automation der Allpass-Koeffizienten über duale Zustandsbänke mit kubischem Smoothstep-Crossfade und robuster C1-Zustandsvererbung selbst bei überlappendem Hochfrequenz-Retriggering.
* **Modulation:**
  32 reziproke Kanten-LFOs mit Hermite-4-Punkt-Fractional-Delay-Interpolation für chorusartige Modenaufweitung ohne Tonhöhen-Drift.

---

## ⚡ SIMD & Performance

* **Header-Only Policy-Architektur:** Wählbares Backend über Template-Policies (`dtqw::simd::SSEBackend`, `dtqw::simd::NEONBackend`, `dtqw::simd::ScalarBackend`).
* **Optimierte Ringpuffer-Guard-Zones:** Zusammenhängende 4-Sample-Loads mit $4 \times 4$-Transposition eliminieren unzusammenhängende Gather-Zugriffe.
* **Echtzeit-Garantien:**
  * **Null Heap-Allokationen** im Audiopfad (`noexcept`).
  * Bitgenaue Pufferinvarianz über Blockgrößen von $1$ bis $2048$ Samples.
  * Vollständig geschützt gegen Denormals und arithmetische Überläufe.

---

## 🛡️ CI-Pipeline & Branch-Protection

Änderungen an der Allpass- und Crossfade-Logik werden durch ein verpflichtendes GitHub-Actions-Gate (`allpass-crossfade-gate`) abgesichert:
* Kompiliert auf **Linux (GCC & Clang, SSE4.1)** und **macOS (AppleClang ARM64 NEON)**.
* Strikte Sanitizer-Prüfung via `-fsanitize=address,undefined` mit `detect_leaks=1`, `strict_init_order=1` und `halt_on_error=1`.
* Automatische Blockierung von Pull Requests bei Memory Corruption, Buffer Overflows oder Undefined Behavior.

### Lokale Verifikation
```bash
./run_sanitizers.sh --force
# oder über CMake:
cmake -B build -DDTQW_ENABLE_SANITIZERS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```
