/**
 * @file DTQWHostAutomationTests.cpp
 * @brief C++17 Testsuite zur Verifikation der klickfreien Host-Automation
 *        der Allpass-Dispersionskoeffizienten im DTQW Reverb.
 *
 * Verifiziert:
 *  1. Klickunterdrückung bei extremen Sprüngen (g = -0.90 -> +0.90) im Vergleich ungesmooth vs. geglättet
 *  2. Sample-genaue lineare Host-Automation (DAW-Puffer-Rampen via setDispersionRamp)
 *  3. Unabhängige 4-Port-Automation (Glättung der 4 Hypercube-Dimensionen)
 *  4. Klickfreies Bypassen/Aktivieren via geglättetem Crossfade (setDispersionEnabled)
 *  5. Akustische Frequenz-Automation (setDispersionFrequency mit bilinearer Transformation)
 *  6. Kontinuierliche schnelle DAW-Automation (50 Hz Modulationskurven, 100k Samples)
 *  7. SIMD-Parität unter aktiver Host-Automation (SSE vs. NEON vs. Skalar > 125 dB SNR)
 *  8. Strikte Allokationsfreiheit (0 Heap-Allokationen) im Audio-Thread unter Automation
 */

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>
#include <atomic>
#include <new>
#include <cstdlib>
#include <algorithm>

// ============================================================================
// Real-Time Allocation Tracker
// ============================================================================
thread_local bool t_allow_allocations = true;
std::atomic<size_t> g_forbidden_allocations{0};

void* operator new(std::size_t size) {
    if (!t_allow_allocations) {
        g_forbidden_allocations.fetch_add(1, std::memory_order_relaxed);
    }
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void* operator new[](std::size_t size) {
    if (!t_allow_allocations) {
        g_forbidden_allocations.fetch_add(1, std::memory_order_relaxed);
    }
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

#include "DTQWReverb.hpp"

// ============================================================================
// Test 1: Click Suppression: Unsmoothed vs Smoothed Extreme Jump
// ============================================================================
void test_click_suppression() {
    std::cout << "[TEST 1] Extreme Jump Click Suppression (Unsmoothed vs Smoothed)... ";

    // Teil A: Isolierte Kanten-Allpasskaskade
    {
        using Vec4 = dtqw::simd::SSEBackend::Vec4;
        dtqw::AllpassCascade4<dtqw::simd::SSEBackend, 2> ap_unsmoothed;
        dtqw::AllpassCascade4<dtqw::simd::SSEBackend, 2> ap_smoothed;
        constexpr size_t N = 4000;
        std::vector<float> in(N);
        for (size_t i = 0; i < N; ++i) in[i] = 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * 440.0f * static_cast<float>(i) / 48000.0f);
        std::vector<float> y_a(N), y_b(N);
        ap_unsmoothed.setCoefficient(-0.85f);
        ap_smoothed.setCoefficient(-0.85f);
        float g_smooth = -0.85f, alpha = 1.0f - std::exp(-1.0f / (0.020f * 48000.0f));
        for (size_t i = 0; i < N; ++i) {
            Vec4 x = dtqw::simd::SSEBackend::set1(in[i]);
            if (i == 2000) ap_unsmoothed.setCoefficient(0.85f);
            Vec4 out_a = ap_unsmoothed.process(x);
            alignas(16) std::array<float, 4> a_arr;
            dtqw::simd::SSEBackend::storeu(a_arr.data(), out_a); y_a[i] = a_arr[0];
            if (i >= 2000) { g_smooth += alpha * (0.85f - g_smooth); ap_smoothed.setCoefficient(g_smooth); }
            Vec4 out_b = ap_smoothed.process(x);
            alignas(16) std::array<float, 4> b_arr;
            dtqw::simd::SSEBackend::storeu(b_arr.data(), out_b); y_b[i] = b_arr[0];
        }
        float max_delta_a = 0.0f, max_delta_b = 0.0f;
        for (size_t i = 1995; i < 2010; ++i) {
            max_delta_a = std::max(max_delta_a, std::abs(y_a[i] - y_a[i - 1]));
            max_delta_b = std::max(max_delta_b, std::abs(y_b[i] - y_b[i - 1]));
        }
        float ap_suppression_db = 20.0f * std::log10(max_delta_a / max_delta_b);
        assert(ap_suppression_db > 15.0f);
    }

    // Teil B: Gesamter DTQW Reverb-Feedbackkreis unter Daueranregung
    constexpr double kSampleRate = 48000.0;
    constexpr size_t kNumSamples = 8000, kJumpSample = 3000;
    std::vector<float> input(kNumSamples);
    for (size_t i = 0; i < kNumSamples; ++i) input[i] = 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * 440.0f * static_cast<float>(i) / 48000.0f);
    std::vector<float> out_a(kNumSamples), out_b(kNumSamples);
    {
        dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
        engine.prepare(kSampleRate); engine.setDispersion(-0.85f, true); engine.setDecayTime(2.0f); engine.setDryWet(1.0f);
        for (size_t i = 0; i < kNumSamples; ++i) {
            if (i == kJumpSample) engine.setDispersion(0.85f, true);
            auto [l, r] = engine.processSample(input[i], input[i]); out_a[i] = l;
        }
    }
    {
        dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
        engine.prepare(kSampleRate); engine.setDispersion(-0.85f, true); engine.setDispersionSmoothingTime(30.0f); engine.setDecayTime(2.0f); engine.setDryWet(1.0f);
        for (size_t i = 0; i < kNumSamples; ++i) {
            if (i == kJumpSample) engine.setDispersion(0.85f, false);
            auto [l, r] = engine.processSample(input[i], input[i]); out_b[i] = l;
        }
    }
    float max_deriv_unsmoothed = 0.0f, max_deriv_smoothed = 0.0f;
    for (size_t i = kJumpSample; i < kJumpSample + 2000; ++i) {
        max_deriv_unsmoothed = std::max(max_deriv_unsmoothed, std::abs(out_a[i] - out_a[i - 1]));
        max_deriv_smoothed = std::max(max_deriv_smoothed, std::abs(out_b[i] - out_b[i - 1]));
    }
    float suppression_db = 20.0f * std::log10(max_deriv_unsmoothed / std::max(1e-12f, max_deriv_smoothed));
    if (suppression_db > 10.0f && max_deriv_smoothed < 0.006f) {
        std::cout << "PASSED (isolated allpass >15 dB, full reverb loop " << suppression_db << " dB click suppression; smoothed derivative " << max_deriv_smoothed << ")\n";
    } else {
        std::cout << "FAILED (Suppression: " << suppression_db << " dB)\n"; std::exit(1);
    }
}

// ============================================================================
// Test 2: Sample-Accurate Linear DAW Buffer Ramp (setDispersionRamp)
// ============================================================================
void test_linear_ramp_automation() {
    std::cout << "[TEST 2] Linear DAW Buffer Ramp (setDispersionRamp)... ";

    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDispersion(-0.50f, true);

    constexpr size_t kRampSamples = 256; // z.B. 256 Samples DAW Puffergröße
    engine.setDispersionRamp(0.60f, kRampSamples);

    assert(engine.isDispersionSmoothing() == true);

    float prev_disp = engine.getCurrentDispersion();
    float expected_step = (0.60f - (-0.50f)) / static_cast<float>(kRampSamples);

    for (size_t n = 0; n < kRampSamples; ++n) {
        engine.processSample(0.0f, 0.0f);
        float cur = engine.getCurrentDispersion();
        float step = cur - prev_disp;
        assert(std::abs(step - expected_step) < 1e-4f);
        prev_disp = cur;
    }

    assert(std::abs(engine.getCurrentDispersion() - 0.60f) < 1e-5f);
    assert(engine.isDispersionSmoothing() == false);

    std::cout << "PASSED (256-sample linear buffer ramp: exact target 0.60 reached, step: "
              << expected_step << ")\n";
}

// ============================================================================
// Test 3: Per-Port Independent Dispersion Automation
// ============================================================================
void test_per_port_automation() {
    std::cout << "[TEST 3] Per-Port Independent Dispersion Automation... ";

    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDispersionSmoothingTime(10.0f); // 10ms

    std::array<float, 4> initial_ports = { -0.7f, -0.2f, 0.3f, 0.8f };
    engine.setPortDispersion(initial_ports, true);

    for (size_t k = 0; k < 4; ++k) {
        assert(std::abs(engine.getCurrentPortDispersion(k) - initial_ports[k]) < 1e-5f);
    }

    // Glätten zu neuen Zielen
    std::array<float, 4> target_ports = { 0.5f, 0.1f, -0.4f, -0.8f };
    engine.setPortDispersion(target_ports, false); // Host-Automation geglättet

    assert(engine.isDispersionSmoothing() == true);

    // 30000 Samples durchlaufen (~625 ms)
    for (size_t i = 0; i < 30000; ++i) {
        engine.processSample(0.0f, 0.0f);
    }

    assert(engine.isDispersionSmoothing() == false);
    for (size_t k = 0; k < 4; ++k) {
        assert(std::abs(engine.getCurrentPortDispersion(k) - target_ports[k]) < 1e-4f);
    }

    std::cout << "PASSED (All 4 spatial hypercube dimensions smoothly converged to target ports)\n";
}

// ============================================================================
// Test 4: Click-Free Enable / Bypass Crossfade (setDispersionEnabled)
// ============================================================================
void test_click_free_enable_bypass() {
    std::cout << "[TEST 4] Click-Free Enable / Bypass Crossfade... ";

    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDispersion(0.75f, true);
    engine.setDispersionEnabled(true, true);
    engine.setDecayTime(2.0f);

    constexpr size_t kNumSamples = 6000;
    constexpr size_t kBypassSample = 2000;
    constexpr size_t kReenableSample = 4000;

    std::vector<float> input(kNumSamples, 0.3f); // DC + Signal zur Transientenerkennung
    float max_deriv = 0.0f;
    float prev_out = 0.0f;

    for (size_t i = 0; i < kNumSamples; ++i) {
        if (i == kBypassSample) {
            engine.setDispersionEnabled(false, false); // Geglättetes Bypassen
        }
        if (i == kReenableSample) {
            engine.setDispersionEnabled(true, false); // Geglättetes Wiedereinschalten
        }

        auto [l, r] = engine.processSample(input[i], input[i]);
        if (i > 100) {
            max_deriv = std::max(max_deriv, std::abs(l - prev_out));
        }
        prev_out = l;
    }

    if (max_deriv < 0.08f) {
        std::cout << "PASSED (Max transition derivative: " << max_deriv << ", zero click pop)\n";
    } else {
        std::cout << "FAILED (Max transition derivative too high: " << max_deriv << ")\n";
        std::exit(1);
    }
}

// ============================================================================
// Test 5: Acoustic Frequency Automation (setDispersionFrequency)
// ============================================================================
void test_frequency_automation() {
    std::cout << "[TEST 5] Acoustic Frequency-to-Allpass Mapping Automation... ";

    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDispersionSmoothingTime(15.0f);

    // Automatisierter Sweep von 200 Hz zu 10.000 Hz
    constexpr size_t kSweepSamples = 4000;
    for (size_t i = 0; i < kSweepSamples; ++i) {
        float f = 200.0f + 9800.0f * (static_cast<float>(i) / static_cast<float>(kSweepSamples));
        engine.setDispersionFrequency(f, false);
        engine.processSample(0.1f, 0.1f);
    }

    // Bei 10 kHz: omega = pi * 10000 / 48000 = 0.6545 rad, tan = 0.7673
    // g = (1 - 0.7673)/(1 + 0.7673) = 0.1317
    float expected_g = (1.0f - std::tan(static_cast<float>(M_PI) * 10000.0f / 48000.0f)) /
                       (1.0f + std::tan(static_cast<float>(M_PI) * 10000.0f / 48000.0f));

    // Nach 6000 weiteren Samples muss der geglättete Wert innerhalb der Toleranz konvergieren
    for (size_t i = 0; i < 6000; ++i) {
        engine.processSample(0.0f, 0.0f);
    }

    float final_g = engine.getCurrentDispersion();
    if (std::abs(final_g - expected_g) >= 1e-3f) {
        std::cerr << "frequency final_g=" << final_g << " expected=" << expected_g << " target=" << engine.getDispersion() << "\n";
        std::exit(1);
    }

    std::cout << "PASSED (Smooth frequency sweep 200Hz -> 10kHz reached target g=" << final_g << ")\n";
}

// ============================================================================
// Test 6: Continuous 50 Hz DAW LFO Modulation (100k Samples Stress Test)
// ============================================================================
void test_continuous_lfo_modulation_stress() {
    std::cout << "[TEST 6] Continuous 50 Hz DAW LFO Automation Stress (100k Samples)... ";

    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDecayTime(2.5f);
    engine.setDampingCutoff(6500.0f);
    engine.setDispersionSmoothingTime(5.0f); // Schnelle 5ms Glättung

    constexpr size_t kStressSamples = 100000;
    bool has_nan_inf = false;
    float max_val = 0.0f;

    for (size_t i = 0; i < kStressSamples; ++i) {
        // Kontinuierliche schnelle LFO-Automation über den vollen Bereich [-0.85, +0.85]
        float lfo = 0.85f * std::sin(2.0f * static_cast<float>(M_PI) * 50.0f * static_cast<float>(i) / 48000.0f);
        engine.setDispersion(lfo, false);

        float in_sig = (i % 1000 == 0) ? 0.8f : 0.01f;
        auto [l, r] = engine.processSample(in_sig, in_sig);

        if (std::isnan(l) || std::isnan(r) || std::isinf(l) || std::isinf(r)) {
            has_nan_inf = true;
            break;
        }
        max_val = std::max(max_val, std::max(std::abs(l), std::abs(r)));
    }

    if (!has_nan_inf && max_val < 3.0f) {
        std::cout << "PASSED (100k samples rock-solid, max peak: " << max_val << ", no blowups)\n";
    } else {
        std::cout << "FAILED (NaN/Inf: " << has_nan_inf << ", peak: " << max_val << ")\n";
        std::exit(1);
    }
}

// ============================================================================
// Test 7: SIMD Parity under Active Host Automation (SSE vs NEON vs Scalar)
// ============================================================================
void test_simd_parity_under_automation() {
    std::cout << "[TEST 7] SIMD Parity under Active Host Automation (SSE vs NEON vs Scalar)... ";

    dtqw::DTQWEngine<dtqw::simd::ScalarBackend> engine_scal;
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine_sse;
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(DTQW_TEST_NEON_ON_X86)
    dtqw::DTQWEngine<dtqw::simd::NEONBackend> engine_neon;
#endif

    engine_scal.prepare(48000.0);
    engine_sse.prepare(48000.0);
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(DTQW_TEST_NEON_ON_X86)
    engine_neon.prepare(48000.0);
#endif

    constexpr size_t kSamples = 24000;
    double sum_sq_sig = 0.0;
    double sum_sq_diff_sse = 0.0;
    double max_diff_sse = 0.0;

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(DTQW_TEST_NEON_ON_X86)
    double sum_sq_diff_neon = 0.0;
    double max_diff_neon = 0.0;
#endif

    for (size_t i = 0; i < kSamples; ++i) {
        // Automation alle 64 Samples (typischer Puffer)
        if (i % 64 == 0) {
            float g = 0.70f * std::sin(static_cast<float>(i) * 0.005f);
            engine_scal.setDispersion(g, false);
            engine_sse.setDispersion(g, false);
#if defined(__ARM_NEON) || defined(__aarch64__) || defined(DTQW_TEST_NEON_ON_X86)
            engine_neon.setDispersion(g, false);
#endif
        }

        float in = (i == 0) ? 1.0f : 0.0f;
        auto [ls, rs] = engine_scal.processSample(in, in);
        auto [le, re] = engine_sse.processSample(in, in);

        double diff_l = static_cast<double>(ls - le);
        sum_sq_sig += ls * ls;
        sum_sq_diff_sse += diff_l * diff_l;
        max_diff_sse = std::max(max_diff_sse, std::abs(diff_l));

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(DTQW_TEST_NEON_ON_X86)
        auto [ln, rn] = engine_neon.processSample(in, in);
        double diff_n = static_cast<double>(ls - ln);
        sum_sq_diff_neon += diff_n * diff_n;
        max_diff_neon = std::max(max_diff_neon, std::abs(diff_n));
#endif
    }

    double snr_sse = 10.0 * std::log10(sum_sq_sig / std::max(1e-20, sum_sq_diff_sse));
    assert(snr_sse > 125.0);

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(DTQW_TEST_NEON_ON_X86)
    double snr_neon = 10.0 * std::log10(sum_sq_sig / std::max(1e-20, sum_sq_diff_neon));
    assert(snr_neon > 125.0);
    std::cout << "PASSED (SSE SNR: " << snr_sse << " dB, NEON SNR: " << snr_neon << " dB)\n";
#else
    std::cout << "PASSED (SSE SNR: " << snr_sse << " dB)\n";
#endif
}

// ============================================================================
// Test 8: Zero Heap Allocations in Audio Thread under Host Automation
// ============================================================================
void test_zero_allocations_automation() {
    std::cout << "[TEST 8] Zero Heap Allocations in Audio Thread under Active Automation... ";

    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);

    g_forbidden_allocations.store(0);
    t_allow_allocations = false; // Sperre Heap-Allokationen im Audio-Thread!

    constexpr size_t kNumSamples = 50000;
    for (size_t i = 0; i < kNumSamples; ++i) {
        if (i % 128 == 0) {
            float g = 0.8f * std::cos(static_cast<float>(i) * 0.01f);
            engine.setDispersion(g, false);
            engine.setDispersionRamp(-g, 128);
            engine.setPortDispersion(i % 4, g * 0.5f, false);
            engine.setDispersionFrequency(1000.0f + static_cast<float>(i % 8000), false);
            if (i % 2048 == 0) {
                engine.setDispersionEnabled((i / 2048) % 2 == 0, false);
            }
        }
        engine.processSample(0.1f, -0.1f);
    }

    t_allow_allocations = true;
    size_t allocs = g_forbidden_allocations.load();

    if (allocs == 0) {
        std::cout << "PASSED (Strictly 0 heap allocations across 50,000 samples and automation commands)\n";
    } else {
        std::cout << "FAILED (" << allocs << " illegal allocations detected!)\n";
        std::exit(1);
    }
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "   DTQW Reverb - Dispersion Host Automation & Click Test Suite   \n";
    std::cout << "=================================================================\n";

    test_click_suppression();
    test_linear_ramp_automation();
    test_per_port_automation();
    test_click_free_enable_bypass();
    test_frequency_automation();
    test_continuous_lfo_modulation_stress();
    test_simd_parity_under_automation();
    test_zero_allocations_automation();

    std::cout << "=================================================================\n";
    std::cout << " SUMMARY: ALL 8 HOST AUTOMATION TESTS PASSED SUCCESSFULLY!       \n";
    std::cout << "=================================================================\n";
    return 0;
}
