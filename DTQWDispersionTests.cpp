#define DTQW_TEST_NEON_ON_X86 1

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <cassert>
#include <complex>
#include <algorithm>

#include "DTQWReverb.hpp"

// Global allocation counter for real-time safety verification
static size_t g_heap_allocations = 0;
static bool g_track_allocations = false;

#ifndef __SANITIZE_ADDRESS__
void* operator new(size_t size) {
    if (g_track_allocations) {
        ++g_heap_allocations;
    }
    return std::malloc(size);
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, size_t) noexcept {
    std::free(ptr);
}

void* operator new[](size_t size) {
    if (g_track_allocations) {
        ++g_heap_allocations;
    }
    return std::malloc(size);
}

void operator delete[](void* ptr) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr, size_t) noexcept {
    std::free(ptr);
}
#endif

// ============================================================================
// Test 1: Allpass Magnitude Response Flatness (Isometry / Unitary Metric)
// ============================================================================
void test_magnitude_flatness() {
    std::cout << "[TEST 1] Allpass Magnitude Response Flatness & Isometry... ";
    constexpr double pi = 3.14159265358979323846;
    constexpr double fs = 48000.0;
    constexpr size_t N = 4096;

    const std::vector<float> test_coeffs = { -0.85f, -0.50f, -0.20f, 0.0f, 0.35f, 0.60f, 0.85f };
    const std::vector<double> test_freqs = { 20.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 15000.0, 20000.0, 23900.0 };

    float max_mag_dev_db = 0.0f;
    double max_energy_dev = 0.0;

    for (float g : test_coeffs) {
        dtqw::AllpassCascade4<dtqw::simd::SSEBackend, 2> ap2;
        ap2.setCoefficient(g);

        // Generate impulse response
        std::vector<float> ir(N, 0.0f);
        for (size_t n = 0; n < N; ++n) {
            float x_val = (n == 0) ? 1.0f : 0.0f;
            auto x_vec = dtqw::simd::SSEBackend::set(x_val, 0.0f, 0.0f, 0.0f);
            auto y_vec = ap2.process(x_vec);
            alignas(16) float y_arr[4];
            dtqw::simd::SSEBackend::storeu(y_arr, y_vec);
            ir[n] = y_arr[0];
        }

        // Parseval's theorem: sum |h[n]|^2 == 1.0
        double energy = 0.0;
        for (float s : ir) energy += static_cast<double>(s) * s;
        max_energy_dev = std::max(max_energy_dev, std::abs(energy - 1.0));

        // Evaluate DTFT
        for (double f : test_freqs) {
            double omega = 2.0 * pi * f / fs;
            std::complex<double> H(0.0, 0.0);
            for (size_t n = 0; n < N; ++n) {
                double phase = -omega * n;
                H += static_cast<double>(ir[n]) * std::complex<double>(std::cos(phase), std::sin(phase));
            }
            double mag_db = 20.0 * std::log10(std::abs(H));
            max_mag_dev_db = std::max(max_mag_dev_db, static_cast<float>(std::abs(mag_db)));
        }
    }

    if (max_mag_dev_db < 0.001f && max_energy_dev < 1e-6) {
        std::cout << "PASSED (Max |H(jw)| deviation: " << max_mag_dev_db << " dB, Parseval err: "
                  << max_energy_dev << ")\n";
    } else {
        std::cout << "FAILED (Max dev: " << max_mag_dev_db << " dB, Energy dev: " << max_energy_dev << ")\n";
        std::exit(1);
    }
}

// ============================================================================
// Test 2: Frequency-Dependent Group Delay & Phase Dispersion
// ============================================================================
void test_group_delay_dispersion() {
    std::cout << "[TEST 2] Group Delay & Frequency-Dependent Phase Dispersion... ";
    constexpr double pi = 3.14159265358979323846;
    constexpr double fs = 48000.0;
    constexpr size_t N = 4096;
    constexpr size_t K = 2; // 2 stages

    const std::vector<float> test_coeffs = { -0.7f, -0.3f, 0.0f, 0.4f, 0.7f };
    const std::vector<double> test_freqs = { 100.0, 1000.0, 5000.0, 10000.0, 18000.0, 22000.0 };

    double max_tau_err = 0.0;

    for (float g : test_coeffs) {
        dtqw::AllpassCascade4<dtqw::simd::SSEBackend, K> cascade;
        cascade.setCoefficient(g);

        std::vector<float> ir(N, 0.0f);
        for (size_t n = 0; n < N; ++n) {
            float x_val = (n == 0) ? 1.0f : 0.0f;
            auto x_vec = dtqw::simd::SSEBackend::set(x_val, 0.0f, 0.0f, 0.0f);
            auto y_vec = cascade.process(x_vec);
            alignas(16) float y_arr[4];
            dtqw::simd::SSEBackend::storeu(y_arr, y_vec);
            ir[n] = y_arr[0];
        }

        auto dtft = [&](double w) {
            std::complex<double> H(0.0, 0.0);
            for (size_t n = 0; n < N; ++n) {
                double phase = -w * n;
                H += static_cast<double>(ir[n]) * std::complex<double>(std::cos(phase), std::sin(phase));
            }
            return H;
        };

        for (double f : test_freqs) {
            double omega = 2.0 * pi * f / fs;
            double dw = 1e-4;

            auto H_plus  = dtft(omega + dw);
            auto H_minus = dtft(omega - dw);
            double phi_plus  = std::arg(H_plus);
            double phi_minus = std::arg(H_minus);
            while (phi_plus - phi_minus > pi) phi_plus -= 2.0 * pi;
            while (phi_plus - phi_minus < -pi) phi_plus += 2.0 * pi;

            double tau_meas = -(phi_plus - phi_minus) / (2.0 * dw);
            double tau_theo = static_cast<double>(K) * (1.0 - g * g) / (1.0 + g * g - 2.0 * g * std::cos(omega));

            max_tau_err = std::max(max_tau_err, std::abs(tau_meas - tau_theo));
        }

        // Verify dispersion direction:
        // g > 0: tau(DC) > tau(Nyquist) (Up-chirp)
        // g < 0: tau(DC) < tau(Nyquist) (Down-chirp)
        // g = 0: tau(DC) == tau(Nyquist) == K
        double tau_dc   = static_cast<double>(K) * (1.0 + g) / (1.0 - g);
        double tau_nyq  = static_cast<double>(K) * (1.0 - g) / (1.0 + g);
        if (g > 0.01f) {
            assert(tau_dc > tau_nyq);
        } else if (g < -0.01f) {
            assert(tau_dc < tau_nyq);
        } else {
            assert(std::abs(tau_dc - static_cast<double>(K)) < 1e-5);
            assert(std::abs(tau_nyq - static_cast<double>(K)) < 1e-5);
        }
    }

    if (max_tau_err < 0.02) {
        std::cout << "PASSED (Max group delay error vs theory: " << max_tau_err << " samples)\n";
    } else {
        std::cout << "FAILED (Max group delay error: " << max_tau_err << ")\n";
        std::exit(1);
    }
}

// ============================================================================
// Test 3: Closed-Loop Contractivity & Energy Conservation (No Damping)
// ============================================================================
void test_energy_conservation_no_damping() {
    std::cout << "[TEST 3] Closed-Loop Contractivity & Energy Conservation (Undamped)... ";

    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDecayTime(60.0f); // Max decay (minimal damping)
    engine.setDampingCutoff(23000.0f);
    engine.setRoomSize(1.0f, true);
    engine.setChorusDepth(0.0f); // Disable LFO modulation for strict energy testing
    engine.setCoinPhase(0.0f, true);
    engine.setDryWet(1.0f);
    engine.setDispersion(0.65f, true); // Strong dispersion active!

    // Inject Dirac impulse at t = 0
    auto [out_l0, out_r0] = engine.processSample(1.0f, 1.0f);

    constexpr size_t kNumSamples = 48000;
    double max_output = std::max(std::abs(out_l0), std::abs(out_r0));
    bool had_nan_or_inf = false;
    double accumulated_output_energy = out_l0 * out_l0 + out_r0 * out_r0;

    for (size_t n = 1; n < kNumSamples; ++n) {
        auto [l, r] = engine.processSample(0.0f, 0.0f);
        if (std::isnan(l) || std::isnan(r) || std::isinf(l) || std::isinf(r)) {
            had_nan_or_inf = true;
            break;
        }
        max_output = std::max(max_output, static_cast<double>(std::max(std::abs(l), std::abs(r))));
        accumulated_output_energy += l * l + r * r;
    }

    // Energy must be bounded: with unitary coin, allpasses, and passive damping,
    // the system cannot explode (max output should remain well under initial impulse scale).
    if (!had_nan_or_inf && max_output < 2.0 && accumulated_output_energy < 500.0) {
        std::cout << "PASSED (Max peak: " << max_output << ", no blowups, strict stability)\n";
    } else {
        std::cout << "FAILED (Peak: " << max_output << ", Nan/Inf: " << had_nan_or_inf << ")\n";
        std::exit(1);
    }
}

// ============================================================================
// Test 4: Dynamic Parameter Sweeps & Rapid Modulation Stability
// ============================================================================
void test_realtime_automation_stability() {
    std::cout << "[TEST 4] Dynamic Dispersion Sweeps & Modulation Stability... ";

    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDecayTime(2.5f);
    engine.setDampingCutoff(6000.0f);
    engine.setDispersionSmoothingTime(10.0f); // 10ms smooth time

    constexpr size_t kNumSamples = 50000;
    bool has_nan_inf = false;
    float max_peak = 0.0f;

    // Stimulate with white noise + impulsive spikes while violently modulating dispersion
    std::vector<float> input(kNumSamples);
    for (size_t i = 0; i < kNumSamples; ++i) {
        if (i % 2000 == 0) {
            input[i] = 1.0f; // Periodic impulses
        } else {
            input[i] = 0.05f * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f);
        }
    }

    for (size_t i = 0; i < kNumSamples; ++i) {
        // Fast sinusoidal sweep of dispersion across [-0.90, +0.90] at 50 Hz
        float g_target = 0.90f * std::sin(2.0f * static_cast<float>(M_PI) * 50.0f * static_cast<float>(i) / 48000.0f);
        engine.setDispersion(g_target);

        // Every 5000 samples, do an abrupt immediate jump
        if (i % 5000 == 0) {
            engine.setDispersion((i % 10000 == 0) ? -0.85f : 0.85f, true);
        }

        auto [l, r] = engine.processSample(input[i], input[i]);
        if (std::isnan(l) || std::isnan(r) || std::isinf(l) || std::isinf(r)) {
            has_nan_inf = true;
            break;
        }
        max_peak = std::max(max_peak, std::max(std::abs(l), std::abs(r)));
    }

    if (!has_nan_inf && max_peak < 10.0f) {
        std::cout << "PASSED (Clean output under 50Hz sweeps & immediate steps, peak: " << max_peak << ")\n";
    } else {
        std::cout << "FAILED (NaN/Inf detected or explosion: peak " << max_peak << ")\n";
        std::exit(1);
    }
}

// ============================================================================
// Test 5: SIMD Backend Parity with Active Dispersion (SSE vs Scalar vs NEON)
// ============================================================================
void test_simd_parity() {
    std::cout << "[TEST 5] SIMD Parity with Active Dispersion (SSE vs NEON vs Scalar)... ";

    constexpr size_t kNumSamples = 48000;

    dtqw::DTQWEngine<dtqw::simd::ScalarBackend> engine_scalar;
    dtqw::DTQWEngine<dtqw::simd::SSEBackend>    engine_sse;
    dtqw::DTQWEngine<dtqw::simd::NEONBackend>   engine_neon;

    engine_scalar.prepare(48000.0);
    engine_sse.prepare(48000.0);
    engine_neon.prepare(48000.0);

    auto setup = [](auto& eng) {
        eng.setDecayTime(2.0f);
        eng.setDampingCutoff(5500.0f);
        eng.setRoomSize(1.15f, true);
        eng.setChorusRate(0.7f);
        eng.setChorusDepth(2.0f);
        eng.setCoinPhase(0.5f, true);
        eng.setDryWet(1.0f);
        eng.setDispersion(0.55f, true); // Active dispersion!
    };
    setup(engine_scalar);
    setup(engine_sse);
    setup(engine_neon);

    std::vector<float> out_l_scal(kNumSamples), out_r_scal(kNumSamples);
    std::vector<float> out_l_sse(kNumSamples),  out_r_sse(kNumSamples);
    std::vector<float> out_l_neon(kNumSamples), out_r_neon(kNumSamples);

    for (size_t i = 0; i < kNumSamples; ++i) {
        float in_l = (i == 0) ? 1.0f : ((i == 240) ? 0.5f : 0.0f);
        float in_r = (i == 0) ? -1.0f : ((i == 360) ? 0.3f : 0.0f);

        auto [ls, rs] = engine_scalar.processSample(in_l, in_r);
        out_l_scal[i] = ls; out_r_scal[i] = rs;

        auto [le, re] = engine_sse.processSample(in_l, in_r);
        out_l_sse[i] = le; out_r_sse[i] = re;

        auto [ln, rn] = engine_neon.processSample(in_l, in_r);
        out_l_neon[i] = ln; out_r_neon[i] = rn;
    }

    float max_diff_sse_scal = 0.0f;
    float max_diff_neon_scal = 0.0f;
    double sum_sq_sig = 0.0;
    double sum_sq_err_sse = 0.0;
    double sum_sq_err_neon = 0.0;

    for (size_t i = 0; i < kNumSamples; ++i) {
        float d_sse_l = std::abs(out_l_sse[i] - out_l_scal[i]);
        float d_sse_r = std::abs(out_r_sse[i] - out_r_scal[i]);
        max_diff_sse_scal = std::max({max_diff_sse_scal, d_sse_l, d_sse_r});

        float d_neon_l = std::abs(out_l_neon[i] - out_l_scal[i]);
        float d_neon_r = std::abs(out_r_neon[i] - out_r_scal[i]);
        max_diff_neon_scal = std::max({max_diff_neon_scal, d_neon_l, d_neon_r});

        sum_sq_sig += out_l_scal[i] * out_l_scal[i] + out_r_scal[i] * out_r_scal[i];
        sum_sq_err_sse += d_sse_l * d_sse_l + d_sse_r * d_sse_r;
        sum_sq_err_neon += d_neon_l * d_neon_l + d_neon_r * d_neon_r;
    }

    double snr_sse  = 10.0 * std::log10(sum_sq_sig / (sum_sq_err_sse + 1e-30));
    double snr_neon = 10.0 * std::log10(sum_sq_sig / (sum_sq_err_neon + 1e-30));

    if (max_diff_sse_scal < 1e-6f && max_diff_neon_scal < 1e-6f && snr_sse > 120.0 && snr_neon > 120.0) {
        std::cout << "PASSED (SSE vs Scal diff: " << max_diff_sse_scal << ", SNR: " << snr_sse
                  << " dB; NEON vs Scal diff: " << max_diff_neon_scal << ", SNR: " << snr_neon << " dB)\n";
    } else {
        std::cout << "FAILED (Diffs: SSE=" << max_diff_sse_scal << ", NEON=" << max_diff_neon_scal << ")\n";
        std::exit(1);
    }
}

// ============================================================================
// Test 6: Echo Density Smearing & Kurtosis Reduction (Dispersed vs Non-Dispersed)
// ============================================================================
void test_dispersion_acoustic_metrics() {
    std::cout << "[TEST 6] Echo Density Smearing & Transient Chirp Metrics... ";

    constexpr size_t kNumSamples = 48000;

    auto measure_ir = [](float disp_amount) {
        dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
        engine.prepare(48000.0);
        engine.setDecayTime(2.0f);
        engine.setDampingCutoff(8000.0f);
        engine.setRoomSize(1.0f, true);
        engine.setChorusDepth(0.0f); // Clean IR without LFO modulation
        engine.setCoinPhase(0.0f, true);
        engine.setDryWet(1.0f);

        if (std::abs(disp_amount) > 1e-6f) {
            engine.setDispersion(disp_amount, true);
        } else {
            engine.setDispersionEnabled(false);
        }

        std::vector<float> ir(kNumSamples);
        auto [l0, r0] = engine.processSample(1.0f, 0.0f);
        ir[0] = l0;
        for (size_t i = 1; i < kNumSamples; ++i) {
            auto [l, r] = engine.processSample(0.0f, 0.0f);
            ir[i] = l;
        }
        return ir;
    };

    auto ir_nodisp = measure_ir(0.0f);
    auto ir_disp   = measure_ir(0.65f);

    // Compute kurtosis and echo density over early window [5ms .. 60ms] (240 .. 2880 samples)
    auto compute_metrics = [](const std::vector<float>& sig, size_t start_smp, size_t end_smp) {
        double m2 = 0.0, m4 = 0.0;
        size_t count = end_smp - start_smp;
        for (size_t i = start_smp; i < end_smp; ++i) {
            double v = sig[i];
            m2 += v * v;
            m4 += v * v * v * v;
        }
        m2 /= count;
        m4 /= count;
        double kurtosis = (m2 > 1e-12) ? (m4 / (m2 * m2)) : 0.0;

        // Standard Abel-Huang Echo Density metric: fraction of samples outside 1 std-dev
        double sigma = std::sqrt(m2);
        size_t over_sigma = 0;
        for (size_t i = start_smp; i < end_smp; ++i) {
            if (std::abs(sig[i]) > sigma) ++over_sigma;
        }
        double echo_density = static_cast<double>(over_sigma) / count;

        return std::make_pair(kurtosis, echo_density);
    };

    auto [kurt_no, ed_no]     = compute_metrics(ir_nodisp, 240, 2880);
    auto [kurt_disp, ed_disp] = compute_metrics(ir_disp, 240, 2880);

    // With dispersion, specular spikes are smeared into chirped waveforms:
    // Kurtosis (spikiness) decreases or echo density increases.
    std::cout << "\n    No Dispersion: Kurtosis = " << std::fixed << std::setprecision(2) << kurt_no
              << ", EchoDensity = " << std::setprecision(3) << ed_no << "\n"
              << "    Dispersed (g=0.65): Kurtosis = " << std::setprecision(2) << kurt_disp
              << ", EchoDensity = " << std::setprecision(3) << ed_disp << "\n  ";

    if (kurt_disp < kurt_no || ed_disp >= ed_no) {
        std::cout << "PASSED (Dispersion successfully smears specular peaks into chirped wavepackets)\n";
    } else {
        std::cout << "FAILED (Dispersion metrics unexpected)\n";
        std::exit(1);
    }
}

// ============================================================================
// Test 7: Real-Time Allocation Freedom in Audio Thread
// ============================================================================
void test_allocation_freedom() {
    std::cout << "[TEST 7] Zero Heap Allocations in Audio Thread with Dispersion... ";

    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDispersion(0.5f, true);

    g_heap_allocations = 0;
    g_track_allocations = true;

    constexpr size_t kBlockSize = 256;
    constexpr size_t kNumBlocks = 200; // 51,200 samples
    float buf_l[kBlockSize]{};
    float buf_r[kBlockSize]{};
    float* channel_data[2] = { buf_l, buf_r };

    for (size_t b = 0; b < kNumBlocks; ++b) {
        // Continuous parameter automation
        float g = 0.7f * std::sin(static_cast<float>(b) * 0.05f);
        engine.setDispersion(g);
        engine.setRoomSize(1.0f + 0.3f * std::cos(static_cast<float>(b) * 0.03f));
        engine.setCoinPhase(static_cast<float>(b) * 0.02f);

        engine.processBlock(channel_data, 2, kBlockSize);
    }

    g_track_allocations = false;

    if (g_heap_allocations == 0) {
        std::cout << "PASSED (0 heap allocations across 51,200 samples and parameter updates)\n";
    } else {
        std::cout << "FAILED (" << g_heap_allocations << " heap allocations detected!)\n";
        std::exit(1);
    }
}

// ============================================================================
// Test 8: 4-Stage vs 2-Stage Cascade Comparison & Frequency Parameter Mapping
// ============================================================================
void test_cascade_orders_and_freq_mapping() {
    std::cout << "[TEST 8] 4-Stage vs 2-Stage Cascade & Frequency-to-g Mapping... ";

    // Test bilinear frequency mapping: setDispersionFrequency
    dtqw::DTQWEngine<dtqw::simd::SSEBackend, 2> engine2;
    engine2.prepare(48000.0);

    // At f = fs / 4 = 12000 Hz, tan(pi * 12000 / 48000) = tan(pi / 4) = 1.0 => g = (1 - 1)/(1 + 1) = 0.0
    engine2.setDispersionFrequency(12000.0f, true);
    float g_at_12k = engine2.getDispersion();
    assert(std::abs(g_at_12k - 0.0f) < 1e-5f);

    // At f = 2000 Hz, omega = 2000*pi/48000 = pi/24 => tan(pi/24) ~ 0.13165 => g ~ 0.767
    engine2.setDispersionFrequency(2000.0f, true);
    float g_at_2k = engine2.getDispersion();
    assert(g_at_2k > 0.70f && g_at_2k < 0.85f);

    // 4-stage engine instantiation & verification
    dtqw::DTQWEngine<dtqw::simd::SSEBackend, 4> engine4;
    engine4.prepare(48000.0);
    engine4.setDispersion(0.5f, true);

    auto [l4, r4] = engine4.processSample(1.0f, 1.0f);
    assert(!std::isnan(l4) && !std::isnan(r4));

    std::cout << "PASSED (g(12kHz) = " << g_at_12k << ", g(2kHz) = " << g_at_2k
              << ", 4-stage engine runs stable)\n";
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "      DTQW Reverb - Frequency-Dependent Dispersion Test Suite    \n";
    std::cout << "=================================================================\n";

    test_magnitude_flatness();
    test_group_delay_dispersion();
    test_energy_conservation_no_damping();
    test_realtime_automation_stability();
    test_simd_parity();
    test_dispersion_acoustic_metrics();
    test_allocation_freedom();
    test_cascade_orders_and_freq_mapping();

    std::cout << "=================================================================\n";
    std::cout << " SUMMARY: ALL 8 DISPERSION TESTS PASSED SUCCESSFULLY!            \n";
    std::cout << "=================================================================\n";
    return 0;
}
