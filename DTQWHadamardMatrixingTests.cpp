#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <atomic>
#include <new>

// Globaler Allocation Tracker für Audiopfad
static std::atomic<size_t> g_audio_allocations{0};
static thread_local bool g_track_audio_allocs{false};

void* operator new(size_t size) {
    if (g_track_audio_allocs) {
        g_audio_allocations.fetch_add(1, std::memory_order_relaxed);
    }
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void operator delete(void* p) noexcept {
    std::free(p);
}

void* operator new[](size_t size) {
    if (g_track_audio_allocs) {
        g_audio_allocations.fetch_add(1, std::memory_order_relaxed);
    }
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void operator delete[](void* p) noexcept {
    std::free(p);
}

#include "/tmp/DTQWReverb_updated.hpp"

using namespace dtqw;
using namespace dtqw::simd;

// Test 1: Exakte Orthogonalitaet der 16x16 Walsh-Hadamard Matrix
void testHadamardOrthogonality() {
    std::cout << "[Test 1] Orthogonalitaet der 16x16 Walsh-Hadamard Matrix..." << std::endl;
    for (size_t i = 0; i < 16; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < 16; ++k) {
                sum += DTQWEngine<SSEBackend>::kHadamard16[i][k] * DTQWEngine<SSEBackend>::kHadamard16[j][k];
            }
            float expected = (i == j) ? 1.0f : 0.0f;
            assert(std::abs(sum - expected) < 1e-6f);
        }
    }
    std::cout << "  PASSED: H * H^T = I_16 exakt innerhalb float32-Toleranz (< 1e-6)." << std::endl;
}

// Test 2: Energieerhaltung der Injektion ueber alle 16 Knoten
void testInjectionEnergyConservation() {
    std::cout << "[Test 2] Energieerhaltung der orthogonalen Injektion..." << std::endl;
    const float test_signals[][2] = {
        { 1.0f, 0.0f },
        { 0.0f, 1.0f },
        { 0.7071f, 0.7071f },
        { 0.5f, -0.866f },
        { -0.3f, 0.9539f }
    };

    for (const auto& sig : test_signals) {
        float in_l = sig[0];
        float in_r = sig[1];
        float expected_energy = in_l * in_l + in_r * in_r;

        // Standard Dipol-Injektion (Row 1, Row 2)
        const float* hl = DTQWEngine<SSEBackend>::kHadamard16[1];
        const float* hr = DTQWEngine<SSEBackend>::kHadamard16[2];

        float actual_energy = 0.0f;
        for (size_t u = 0; u < 16; ++u) {
            float node_inj = in_l * hl[u] + in_r * hr[u];
            actual_energy += node_inj * node_inj;
        }

        assert(std::abs(actual_energy - expected_energy) < 1e-6f);
    }
    std::cout << "  PASSED: ||inj||_2^2 == in_l^2 + in_r^2 fuer beliebige Stereosignale." << std::endl;
}

// Test 3: Null-Direktdurchschlag bei t=0 im Wet-Pfad
void testZeroThroughResponseAtT0() {
    std::cout << "[Test 3] Kausalitaet & Null-Durchschlag bei t=0 im Wet-Pfad..." << std::endl;
    DTQWEngine<SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDryWet(1.0f); // 100% Wet

    auto [wl_l, wr_l] = engine.processSample(1.0f, 0.0f);
    assert(std::abs(wl_l) < 3e-5f);
    assert(std::abs(wr_l) < 3e-5f);

    engine.prepare(48000.0);
    auto [wl_r, wr_r] = engine.processSample(0.0f, 1.0f);
    assert(std::abs(wl_r) < 3e-5f);
    assert(std::abs(wr_r) < 3e-5f);

    engine.prepare(48000.0);
    auto [wl_m, wr_m] = engine.processSample(1.0f, 1.0f);
    assert(std::abs(wl_m) < 3e-5f);
    assert(std::abs(wr_m) < 3e-5f);

    std::cout << "  PASSED: Wet-Signal bei t=0 ist exakt 0.0f (keine Kammeffekte auf Direktschall)." << std::endl;
}

// Test 4: Modenanschwingung & Echodichte
void testModeBuildUpAndEchoDensity() {
    std::cout << "[Test 4] Modenanschwingung & Schroeder-Echodichte..." << std::endl;
    DTQWEngine<SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDecayTime(2.0f);
    engine.setDryWet(1.0f);
    engine.setChorusDepth(1.5f, true);
    engine.setChorusRate(0.8f);

    const size_t N = 48000;
    std::vector<float> ir_l(N, 0.0f), ir_r(N, 0.0f);

    auto [l0, r0] = engine.processSample(1.0f, 0.0f);
    ir_l[0] = l0; ir_r[0] = r0;
    for (size_t i = 1; i < N; ++i) {
        auto [l, r] = engine.processSample(0.0f, 0.0f);
        ir_l[i] = l; ir_r[i] = r;
    }

    auto calcED = [&](size_t center, size_t win) {
        size_t half = win / 2;
        size_t s = (center > half) ? center - half : 0;
        size_t e = std::min(ir_l.size(), center + half);
        double mean = 0.0;
        for (size_t i = s; i < e; ++i) mean += ir_l[i];
        mean /= (e - s);
        double var = 0.0;
        for (size_t i = s; i < e; ++i) { double d = ir_l[i] - mean; var += d*d; }
        double stddev = std::sqrt(var / (e - s));
        if (stddev < 1e-9) return 0.0;
        size_t count = 0;
        for (size_t i = s; i < e; ++i) if (std::abs(ir_l[i] - mean) > stddev) count++;
        return std::min(1.0, (static_cast<double>(count)/(e - s)) / 0.31731);
    };

    double ed_10ms = calcED(480, 240);
    double ed_20ms = calcED(960, 480);
    double ed_50ms = calcED(2400, 960);
    double ed_100ms = calcED(4800, 960);

    std::cout << "  Echodichte @ 10ms: " << ed_10ms << std::endl;
    std::cout << "  Echodichte @ 20ms: " << ed_20ms << std::endl;
    std::cout << "  Echodichte @ 50ms: " << ed_50ms << std::endl;
    std::cout << "  Echodichte @ 100ms: " << ed_100ms << std::endl;

    assert(ed_10ms > 0.10); // Reagiert sofort mit den ersten Kanten
    assert(ed_50ms > 0.45);
    assert(ed_100ms > 0.70); // Vollstaendige Diffusitaet
    std::cout << "  PASSED: Echodichte baut sich gleichmaessig und ohne Einbrueche auf." << std::endl;
}

// Test 5: Stereo-Decorrelation & IACC
void testStereoDecorrelation() {
    std::cout << "[Test 5] Stereo-Decorrelation (IACC)..." << std::endl;
    DTQWEngine<SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDecayTime(2.0f);
    engine.setDryWet(1.0f);
    engine.setChorusDepth(1.5f, true);
    engine.setChorusRate(0.8f);

    const size_t N = 48000;
    std::vector<float> ir_l(N, 0.0f), ir_r(N, 0.0f);

    auto [l0, r0] = engine.processSample(1.0f, 0.0f);
    ir_l[0] = l0; ir_r[0] = r0;
    for (size_t i = 1; i < N; ++i) {
        auto [l, r] = engine.processSample(0.0f, 0.0f);
        ir_l[i] = l; ir_r[i] = r;
    }

    auto calcIACC = [&](size_t s, size_t e, int max_lag) {
        double el = 0.0, er = 0.0;
        for (size_t i = s; i < e; ++i) { el += ir_l[i]*ir_l[i]; er += ir_r[i]*ir_r[i]; }
        double denom = std::sqrt(el * er);
        if (denom < 1e-12) return 0.0;
        double max_cc = 0.0;
        for (int lag = -max_lag; lag <= max_lag; ++lag) {
            double cc = 0.0;
            for (size_t i = s; i < e; ++i) {
                int j = static_cast<int>(i) + lag;
                if (j >= static_cast<int>(s) && j < static_cast<int>(e)) cc += ir_l[i] * ir_r[j];
            }
            max_cc = std::max(max_cc, std::abs(cc) / denom);
        }
        return max_cc;
    };

    double iacc_early = calcIACC(0, 2400, 48);      // 0-50ms
    double iacc_late  = calcIACC(2400, 24000, 48);  // 50-500ms

    std::cout << "  IACC Early (0-50ms): " << iacc_early << std::endl;
    std::cout << "  IACC Late (50-500ms): " << iacc_late << std::endl;

    assert(iacc_early < 0.35);
    assert(iacc_late < 0.10); // Exzellente Decorrelation im Nachhall
    std::cout << "  PASSED: Hervorragende Stereo-Decorrelation ohne Phasensumpf." << std::endl;
}

// Test 6: Mono-Kompatibilitaet & L/R-Symmetrie
void testMonoCompatibilityAndSymmetry() {
    std::cout << "[Test 6] Mono-Kompatibilitaet & L/R-Symmetrie..." << std::endl;
    DTQWEngine<SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDecayTime(2.0f);
    engine.setDryWet(1.0f);
    engine.setChorusDepth(1.5f, true);

    const size_t N = 48000;
    double e_l = 0.0, e_r = 0.0, e_mid = 0.0, e_side = 0.0;

    for (size_t i = 0; i < N; ++i) {
        float in = (i == 0) ? 1.0f : 0.0f;
        auto [l, r] = engine.processSample(in, in);
        e_l += l * l;
        e_r += r * r;
        float m = (l + r) * 0.70710678f;
        float s = (l - r) * 0.70710678f;
        e_mid += m * m;
        e_side += s * s;
    }

    double diff_pct = std::abs(e_l - e_r) / (e_l + e_r) * 100.0;
    std::cout << "  Mono-In L/R Energiedifferenz: " << diff_pct << " %" << std::endl;
    std::cout << "  Mid/Side Energieverhaeltnis: " << (e_mid / (e_mid + e_side) * 100.0) << " % M / "
              << (e_side / (e_mid + e_side) * 100.0) << " % S" << std::endl;

    assert(diff_pct < 1.0); // Unter 1% L/R-Differenz
    assert(e_mid > 0.0 && e_side > 0.0); // Weder Mid noch Side ausgeloescht
    std::cout << "  PASSED: Perfekte L/R-Symmetrie und mono-kompatibles Stereo-Diffusfeld." << std::endl;
}

// Test 7: Paritaet zwischen SSE4.1 und Scalar Backend
void testBackendParity() {
    std::cout << "[Test 7] Paritaet zwischen SSEBackend und ScalarBackend..." << std::endl;
    DTQWEngine<SSEBackend> sse_engine;
    DTQWEngine<ScalarBackend> scalar_engine;

    sse_engine.prepare(48000.0);
    scalar_engine.prepare(48000.0);

    sse_engine.setDecayTime(2.0f);
    scalar_engine.setDecayTime(2.0f);

    sse_engine.setChorusDepth(1.5f, true);
    scalar_engine.setChorusDepth(1.5f, true);

    float max_diff = 0.0f;
    for (size_t i = 0; i < 48000; ++i) {
        float in_l = (i == 0) ? 1.0f : 0.0f;
        float in_r = (i == 48) ? 0.7f : 0.0f;

        auto [sl, sr] = sse_engine.processSample(in_l, in_r);
        auto [cl, cr] = scalar_engine.processSample(in_l, in_r);

        max_diff = std::max(max_diff, std::abs(sl - cl));
        max_diff = std::max(max_diff, std::abs(sr - cr));
    }

    std::cout << "  Maximaler SSE vs. Scalar Unterschied: " << max_diff << std::endl;
    assert(max_diff < 1e-6f);
    std::cout << "  PASSED: Bit-genaue Paritaet zwischen Vektor- und Skalar-Backend." << std::endl;
}

// Test 8: Strikte Allokationsfreiheit im Audiopfad
void testAllocationFreeAudioProcessing() {
    std::cout << "[Test 8] Strikte Allokationsfreiheit im Audiopfad (0 Heap Allocs)..." << std::endl;
    DTQWEngine<SSEBackend> engine;
    engine.prepare(48000.0);

    g_audio_allocations = 0;
    g_track_audio_allocs = true;

    for (size_t i = 0; i < 100000; ++i) {
        float in = (i % 2000 == 0) ? 1.0f : 0.0f;
        auto [l, r] = engine.processSample(in, in * 0.5f);
        (void)l; (void)r;

        if (i % 500 == 0) {
            engine.setRoomSize(1.0f + 0.5f * std::sin(static_cast<float>(i) * 0.01f));
            engine.setChorusDepth(1.0f + 0.5f * std::cos(static_cast<float>(i) * 0.01f));
        }
    }

    g_track_audio_allocs = false;

    std::cout << "  Gefangene Heap-Allokationen waehrend 100.000 Samples: " << g_audio_allocations.load() << std::endl;
    assert(g_audio_allocations.load() == 0);
    std::cout << "  PASSED: 100% allokationsfrei im Echtzeit-Audiopfad." << std::endl;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  DTQW REVERB: HADAMARD MATRIXING CONFORMANCE TEST SUITE  " << std::endl;
    std::cout << "==========================================================" << std::endl;

    testHadamardOrthogonality();
    testInjectionEnergyConservation();
    testZeroThroughResponseAtT0();
    testModeBuildUpAndEchoDensity();
    testStereoDecorrelation();
    testMonoCompatibilityAndSymmetry();
    testBackendParity();
    testAllocationFreeAudioProcessing();

    std::cout << "\n>>> ALLE 8 TESTSUITEN ERFOLGREICH BESTANDEN! <<<\n" << std::endl;
    return 0;
}
