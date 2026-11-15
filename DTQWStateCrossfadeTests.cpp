#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <new>
#include <atomic>

// Global allocation tracker
static std::atomic<size_t> g_audio_thread_allocations{0};
static std::atomic<bool> g_track_allocations{false};

void* operator new(std::size_t size) {
    if (g_track_allocations.load(std::memory_order_relaxed)) {
        g_audio_thread_allocations.fetch_add(1, std::memory_order_relaxed);
    }
    return std::malloc(size);
}

void* operator new[](std::size_t size) {
    if (g_track_allocations.load(std::memory_order_relaxed)) {
        g_audio_thread_allocations.fetch_add(1, std::memory_order_relaxed);
    }
    return std::malloc(size);
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__aarch64__) && !defined(__ARM_NEON)
#define DTQW_TEST_NEON_ON_X86 1
#endif
#include "DTQWReverb.hpp"

// ============================================================================
// Test 1: State Inheritance & C1 Continuity at Crossfade Boundary
// ============================================================================
void test_state_inheritance_and_c1_continuity() {
    std::cout << "[TEST 1] State Inheritance & C1 Boundary Continuity... ";
    using Vec4 = dtqw::simd::SSEBackend::Vec4;
    dtqw::AllpassCascade4<dtqw::simd::SSEBackend, 2> ap;
    ap.reset();
    ap.setCoefficient(-0.75f);

    constexpr size_t N = 1000;
    std::vector<float> in(N);
    for (size_t i = 0; i < N; ++i) {
        in[i] = 0.4f * std::sin(2.0f * static_cast<float>(M_PI) * 500.0f * static_cast<float>(i) / 48000.0f);
    }

    std::vector<float> out(N);
    constexpr size_t kCrossfadeStart = 500;
    constexpr size_t kCrossfadeSamples = 480; // 10 ms bei 48 kHz

    for (size_t i = 0; i < N; ++i) {
        if (i == kCrossfadeStart) {
            ap.startCrossfade(0.75f, kCrossfadeSamples);
            assert(ap.isCrossfading());
        }
        Vec4 x = dtqw::simd::SSEBackend::set1(in[i]);
        Vec4 y = ap.process(x);
        alignas(16) std::array<float, 4> y_arr;
        dtqw::simd::SSEBackend::storeu(y_arr.data(), y);
        out[i] = y_arr[0];
    }

    // Prüfe Ableitung genau an der Nahtstelle (i = 500)
    // Bei C1-Glattheit darf die Ableitung an der Grenze nicht schlagartig springen
    float deriv_before = std::abs(out[kCrossfadeStart - 1] - out[kCrossfadeStart - 2]);
    float deriv_boundary = std::abs(out[kCrossfadeStart] - out[kCrossfadeStart - 1]);
    float deriv_after = std::abs(out[kCrossfadeStart + 1] - out[kCrossfadeStart]);

    // Die Ableitungsdifferenz an der Grenze muss innerhalb normaler Signaloszillation liegen
    float delta_deriv = std::abs(deriv_boundary - deriv_before);
    assert(delta_deriv < 0.005f);
    std::cout << "PASSED (Boundary delta derivative: " << delta_deriv << ", C1 smooth transition)\n";
}

// ============================================================================
// Test 2: Timbre & Dispersion Ring-Out Smoothness (Jump vs State Crossfade)
// ============================================================================
void test_timbre_ringout_smoothness() {
    std::cout << "[TEST 2] Timbre & Dispersion Ring-Out Smoothness (Hard Jump vs State Crossfade)... ";
    constexpr double kSampleRate = 48000.0;
    constexpr size_t kNumSamples = 8000;
    constexpr size_t kChangeSample = 3000;

    std::vector<float> input(kNumSamples);
    for (size_t i = 0; i < kNumSamples; ++i) {
        input[i] = 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * 440.0f * static_cast<float>(i) / 48000.0f);
    }

    std::vector<float> out_hard(kNumSamples);
    std::vector<float> out_crossfade(kNumSamples);

    // Hard Jump
    {
        dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
        engine.prepare(kSampleRate);
        engine.setDispersion(-0.85f, true);
        engine.setDecayTime(2.0f);
        engine.setDryWet(1.0f);
        for (size_t i = 0; i < kNumSamples; ++i) {
            if (i == kChangeSample) {
                engine.setDispersion(0.85f, true); // Hard jump
            }
            auto [l, r] = engine.processSample(input[i], input[i]);
            out_hard[i] = l;
        }
    }

    // Zustandsübergreifender Crossfade (25 ms)
    {
        dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
        engine.prepare(kSampleRate);
        engine.setDispersion(-0.85f, true);
        engine.setDispersionCrossfadeTime(25.0f);
        engine.setDispersionCrossfadeEnabled(true);
        engine.setDecayTime(2.0f);
        engine.setDryWet(1.0f);
        for (size_t i = 0; i < kNumSamples; ++i) {
            if (i == kChangeSample) {
                engine.setDispersion(0.85f, false); // State crossfade
            }
            auto [l, r] = engine.processSample(input[i], input[i]);
            out_crossfade[i] = l;
        }
    }

    float max_d_hard = 0.0f, max_d_fade = 0.0f;
    for (size_t i = kChangeSample; i < kChangeSample + 2000; ++i) {
        max_d_hard = std::max(max_d_hard, std::abs(out_hard[i] - out_hard[i - 1]));
        max_d_fade = std::max(max_d_fade, std::abs(out_crossfade[i] - out_crossfade[i - 1]));
    }

    float suppression_db = 20.0f * std::log10(max_d_hard / std::max(1e-12f, max_d_fade));
    assert(suppression_db > 10.0f);
    assert(max_d_fade < 0.005f);

    std::cout << "PASSED (Suppression: " << suppression_db << " dB, max derivative: " << max_d_fade << ")\n";
}

// ============================================================================
// Test 3: Multi-Port Independent State Crossfades
// ============================================================================
void test_multi_port_crossfade() {
    std::cout << "[TEST 3] Multi-Port Independent State Crossfades... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDispersionCrossfadeTime(20.0f); // 20 ms Crossfade (960 Samples)

    std::array<float, 4> initial_ports = { -0.7f, -0.3f, 0.3f, 0.7f };
    engine.setPortDispersion(initial_ports, true);

    std::array<float, 4> target_ports = { 0.7f, 0.3f, -0.3f, -0.7f };
    engine.setPortDispersion(target_ports, false); // Trigger state crossfade

    assert(engine.isDispersionCrossfading() == true);
    assert(engine.isDispersionSmoothing() == true);

    // 1500 Samples durchlaufen (> 960 Samples Crossfade)
    for (size_t i = 0; i < 1500; ++i) {
        engine.processSample(0.05f, 0.05f);
    }

    assert(engine.isDispersionCrossfading() == false);
    assert(engine.isDispersionSmoothing() == false);

    for (size_t k = 0; k < 4; ++k) {
        assert(std::abs(engine.getCurrentPortDispersion(k) - target_ports[k]) < 1e-4f);
    }

    std::cout << "PASSED (All 4 spatial ports smoothly crossfaded to inverted polarities)\n";
}

// ============================================================================
// Test 4: Rapid Overlapping Crossfades (Stress Retriggering)
// ============================================================================
void test_overlapping_crossfades_stress() {
    std::cout << "[TEST 4] Rapid Overlapping State Crossfades Stress (20k Samples)... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDispersionCrossfadeTime(25.0f); // 25 ms Crossfade

    constexpr size_t N = 20000;
    float max_peak = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        // Alle 80 Samples neuer Zielwert, bevor der 1200-Sample-Crossfade beendet ist!
        if (i % 80 == 0) {
            float g = 0.85f * std::sin(static_cast<float>(i) * 0.05f);
            engine.setDispersion(g, false);
        }
        auto [l, r] = engine.processSample(0.1f, 0.1f);
        assert(!std::isnan(l) && !std::isinf(l));
        assert(!std::isnan(r) && !std::isinf(r));
        max_peak = std::max(max_peak, std::max(std::abs(l), std::abs(r)));
    }

    assert(max_peak < 0.5f);
    std::cout << "PASSED (Rock-solid under continuous overlapping crossfades, max peak: " << max_peak << ")\n";
}

// ============================================================================
// Test 5: Dynamic Crossfade Time Configuration (1 ms to 200 ms)
// ============================================================================
void test_crossfade_time_configuration() {
    std::cout << "[TEST 5] Dynamic Crossfade Time Configuration... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);

    engine.setDispersionCrossfadeTime(10.0f);
    assert(std::abs(engine.getDispersionCrossfadeTime() - 10.0f) < 1e-3f);

    engine.setDispersion(0.0f, true);
    engine.setDispersion(0.5f, false);

    // Bei 10 ms @ 48 kHz = 480 Samples
    for (size_t i = 0; i < 400; ++i) {
        engine.processSample(0.0f, 0.0f);
    }
    assert(engine.isDispersionCrossfading() == true);

    for (size_t i = 0; i < 150; ++i) {
        engine.processSample(0.0f, 0.0f);
    }
    assert(engine.isDispersionCrossfading() == false);
    assert(std::abs(engine.getCurrentDispersion() - 0.5f) < 1e-4f);

    std::cout << "PASSED (10 ms crossfade exact timing confirmed)\n";
}

// ============================================================================
// Test 6: Crossfade Enable/Disable Toggle
// ============================================================================
void test_crossfade_toggle() {
    std::cout << "[TEST 6] Crossfade Mode Toggle (Crossfade vs Exp-Smooth)... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);

    engine.setDispersionCrossfadeEnabled(false);
    assert(engine.isDispersionCrossfadeEnabled() == false);

    engine.setDispersion(0.7f, false);
    // Bei deaktiviertem Crossfade wird dispersion_crossfade_active_ nicht gesetzt
    assert(engine.isDispersionCrossfading() == false);
    // Aber dispersion_smoothing ist aktiv (über exponentielles Glätten)
    assert(engine.isDispersionSmoothing() == true);

    engine.setDispersionCrossfadeEnabled(true);
    assert(engine.isDispersionCrossfadeEnabled() == true);

    std::cout << "PASSED (Clean toggle between state-crossfade and exponential smoothing)\n";
}

// ============================================================================
// Test 7: SIMD Parity under Active State Crossfade (Scalar vs SSE vs NEON)
// ============================================================================
void test_simd_parity_state_crossfade() {
    std::cout << "[TEST 7] SIMD Parity under State Crossfades (SSE vs NEON vs Scalar)... ";
    constexpr double kSampleRate = 48000.0;
    constexpr size_t kNumSamples = 12000;

    std::vector<float> input(kNumSamples);
    for (size_t i = 0; i < kNumSamples; ++i) {
        input[i] = 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * 440.0f * static_cast<float>(i) / 48000.0f);
    }

    std::vector<float> out_scalar_l(kNumSamples), out_sse_l(kNumSamples), out_neon_l(kNumSamples);

    // Scalar
    {
        dtqw::DTQWEngine<dtqw::simd::ScalarBackend> engine;
        engine.prepare(kSampleRate);
        engine.setDispersion(-0.6f, true);
        engine.setDispersionCrossfadeTime(15.0f);
        for (size_t i = 0; i < kNumSamples; ++i) {
            if (i == 2000) engine.setDispersion(0.6f, false);
            if (i == 6000) engine.setPortDispersion({0.4f, -0.4f, 0.7f, -0.7f}, false);
            auto [l, r] = engine.processSample(input[i], input[i]);
            out_scalar_l[i] = l;
        }
    }

    // SSE
    {
        dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
        engine.prepare(kSampleRate);
        engine.setDispersion(-0.6f, true);
        engine.setDispersionCrossfadeTime(15.0f);
        for (size_t i = 0; i < kNumSamples; ++i) {
            if (i == 2000) engine.setDispersion(0.6f, false);
            if (i == 6000) engine.setPortDispersion({0.4f, -0.4f, 0.7f, -0.7f}, false);
            auto [l, r] = engine.processSample(input[i], input[i]);
            out_sse_l[i] = l;
        }
    }

    // NEON
    {
        dtqw::DTQWEngine<dtqw::simd::NEONBackend> engine;
        engine.prepare(kSampleRate);
        engine.setDispersion(-0.6f, true);
        engine.setDispersionCrossfadeTime(15.0f);
        for (size_t i = 0; i < kNumSamples; ++i) {
            if (i == 2000) engine.setDispersion(0.6f, false);
            if (i == 6000) engine.setPortDispersion({0.4f, -0.4f, 0.7f, -0.7f}, false);
            auto [l, r] = engine.processSample(input[i], input[i]);
            out_neon_l[i] = l;
        }
    }

    double diff_sse = 0.0, power_ref = 0.0;
    for (size_t i = 0; i < kNumSamples; ++i) {
        double d = static_cast<double>(out_scalar_l[i] - out_sse_l[i]);
        diff_sse += d * d;
        power_ref += static_cast<double>(out_scalar_l[i] * out_scalar_l[i]);
    }
    float snr_sse = static_cast<float>(10.0 * std::log10(power_ref / std::max(1e-15, diff_sse)));
    assert(snr_sse > 120.0f);

    double diff_neon = 0.0;
    for (size_t i = 0; i < kNumSamples; ++i) {
        double d = static_cast<double>(out_scalar_l[i] - out_neon_l[i]);
        diff_neon += d * d;
    }
    float snr_neon = static_cast<float>(10.0 * std::log10(power_ref / std::max(1e-15, diff_neon)));
    assert(snr_neon > 120.0f);

    std::cout << "PASSED (SSE SNR: " << snr_sse << " dB, NEON SNR: " << snr_neon << " dB)\n";
}

// ============================================================================
// Test 8: Zero Heap Allocations in Audio Thread under Active State Crossfades
// ============================================================================
void test_zero_allocations_state_crossfade() {
    std::cout << "[TEST 8] Zero Heap Allocations in Audio Thread (State Crossfading)... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDispersion(-0.5f, true);
    engine.setDispersionCrossfadeTime(15.0f);

    g_audio_thread_allocations.store(0);
    g_track_allocations.store(true);

    constexpr size_t N = 50000;
    for (size_t i = 0; i < N; ++i) {
        if (i % 500 == 0) {
            float g = 0.8f * std::sin(static_cast<float>(i) * 0.01f);
            engine.setDispersion(g, false);
        }
        if (i % 1200 == 0) {
            engine.setPortDispersion({ -0.4f, 0.4f, -0.6f, 0.6f }, false);
        }
        auto [l, r] = engine.processSample(0.05f, 0.05f);
        (void)l; (void)r;
    }

    g_track_allocations.store(false);
    size_t allocs = g_audio_thread_allocations.load();
    assert(allocs == 0);

    std::cout << "PASSED (0 heap allocations across 50,000 samples and dynamic crossfades)\n";
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "    DTQW Reverb - Dual-State Allpass Crossfade Test Suite        \n";
    std::cout << "=================================================================\n";

    test_state_inheritance_and_c1_continuity();
    test_timbre_ringout_smoothness();
    test_multi_port_crossfade();
    test_overlapping_crossfades_stress();
    test_crossfade_time_configuration();
    test_crossfade_toggle();
    test_simd_parity_state_crossfade();
    test_zero_allocations_state_crossfade();

    std::cout << "=================================================================\n";
    std::cout << " SUMMARY: ALL 8 STATE CROSSFADE TESTS PASSED SUCCESSFULLY!       \n";
    std::cout << "=================================================================\n";
    return 0;
}
