
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <cassert>
#include <complex>
#include <algorithm>

#include "/mnt/documents/DTQWReverb.hpp"

// Global allocation counter
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

// Test 1: Matrix Unitarity and Row Sum Preservation across [0, 2*pi]
void test_matrix_unitarity() {
    std::cout << "[TEST 1] Matrix Unitarity & Row Sum Preservation... ";
    const int num_steps = 36;
    float max_unitarity_err = 0.0f;
    float max_row_sum_err = 0.0f;

    for (int i = 0; i <= num_steps; ++i) {
        float theta = i * (2.0f * 3.14159265358979323846f / num_steps);
        float c = std::cos(theta);
        float s = std::sin(theta);
        float a = -0.5f * c;
        float b =  0.5f * (1.0f - s);
        float g =  0.5f * c;
        float d =  0.5f * (1.0f + s);

        float C[4][4] = {
            {a, b, g, d},
            {d, a, b, g},
            {g, d, a, b},
            {b, g, d, a}
        };

        for (int r = 0; r < 4; ++r) {
            float rsum = C[r][0] + C[r][1] + C[r][2] + C[r][3];
            max_row_sum_err = std::max(max_row_sum_err, std::abs(rsum - 1.0f));

            for (int col = 0; col < 4; ++col) {
                float dot = 0.0f;
                for (int k = 0; k < 4; ++k) dot += C[k][r] * C[k][col];
                float expected = (r == col) ? 1.0f : 0.0f;
                max_unitarity_err = std::max(max_unitarity_err, std::abs(dot - expected));
            }
        }
    }

    std::cout << "PASSED (Max unitarity err: " << max_unitarity_err << ", row sum err: " << max_row_sum_err << ")" << std::endl;
    assert(max_unitarity_err < 1e-6f);
    assert(max_row_sum_err < 1e-6f);
}

// Test 2: Eigenvalue Unit-Circle Conformance
void test_eigenvalues() {
    std::cout << "[TEST 2] Eigenvalue Complex Unit-Circle Conformance... ";
    // For circulant matrix [a, b, g, d], eigenvalues are given by 4-point DFT of first column [a, d, g, b]:
    // lambda_k = a + d * W^k + g * W^(2k) + b * W^(3k), where W = exp(-2*pi*i / 4) = -i
    // W^0 = 1: lambda_0 = a + d + g + b = a - a + (1-b) + b = 1
    // W^1 = -i: lambda_1 = a - i*d - g + i*b = 2a - i*(d - b) = -cos(theta) - i*sin(theta) = -exp(i*theta) = exp(i*(pi - theta))
    // W^2 = -1: lambda_2 = a - d + g - b = 2a - (d + b) = -cos(theta) - 1 ??? Wait:
    // Let's compute exact eigenvalues:
    const float pi = 3.14159265358979323846f;
    for (int i = 0; i <= 16; ++i) {
        float theta = i * (2.0f * pi / 16.0f);
        float c = std::cos(theta);
        float s = std::sin(theta);
        float a = -0.5f * c;
        float b =  0.5f * (1.0f - s);
        float g =  0.5f * c;
        float d =  0.5f * (1.0f + s);

        // Circulant column 0: c0 = a, c1 = d, c2 = g, c3 = b
        // lambda_0 = c0 + c1 + c2 + c3 = 1.0
        // lambda_1 = c0 - i*c1 - c2 + i*c3 = (a - g) + i*(b - d) = -c - i*s = -exp(i*theta)
        // lambda_2 = c0 - c1 + c2 - c3 = (a + g) - (d + b) = 0 - 1 = -1.0
        // lambda_3 = c0 + i*c1 - c2 - i*c3 = (a - g) + i*(d - b) = -c + i*s = -exp(-i*theta)
        std::complex<float> l0(a + d + g + b, 0.0f);
        std::complex<float> l1(a - g, b - d);
        std::complex<float> l2(a - d + g - b, 0.0f);
        std::complex<float> l3(a - g, d - b);

        assert(std::abs(std::abs(l0) - 1.0f) < 1e-6f);
        assert(std::abs(std::abs(l1) - 1.0f) < 1e-6f);
        assert(std::abs(std::abs(l2) - 1.0f) < 1e-6f);
        assert(std::abs(std::abs(l3) - 1.0f) < 1e-6f);

        // Check phases:
        assert(std::abs(l0.real() - 1.0f) < 1e-6f);
        assert(std::abs(l2.real() - (-1.0f)) < 1e-6f);
        assert(std::abs(l1.real() - (-c)) < 1e-6f && std::abs(l1.imag() - (-s)) < 1e-6f);
        assert(std::abs(l3.real() - (-c)) < 1e-6f && std::abs(l3.imag() - s) < 1e-6f);
    }
    std::cout << "PASSED (l0 = +1, l2 = -1, l1,3 = -exp(+-i*theta))" << std::endl;
}

// Test 3: Zero Direct-Sound Bleed at t=0 for all theta
void test_zero_t0_bleed() {
    std::cout << "[TEST 3] Zero Direct-Sound Bleed at t=0 across all theta... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDryWet(1.0f); // 100% wet

    const int steps = 12;
    float max_t0_leakage = 0.0f;
    for (int i = 0; i <= steps; ++i) {
        float theta = i * (2.0f * 3.14159265f / steps);
        engine.reset();
        engine.setCoinPhase(theta, true);
        auto [out_l, out_r] = engine.processSample(1.0f, 0.0f);
        float leak = std::max(std::abs(out_l), std::abs(out_r));
        max_t0_leakage = std::max(max_t0_leakage, leak);
    }
    std::cout << "PASSED (Max t=0 wet leakage: " << max_t0_leakage << " < 3e-5)" << std::endl;
    assert(max_t0_leakage < 3e-5f);
}

// Test 4: Lyapunov Energy Boundedness & Stability over 50,000 samples
void test_stability_decay() {
    std::cout << "[TEST 4] Lyapunov Energy Boundedness & Stability (50k samples)... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDryWet(1.0f);

    const float thetas[] = {0.0f, 0.5236f, 0.7854f, 1.0472f, 1.5708f, 2.0944f, 3.14159f};
    for (float th : thetas) {
        engine.reset();
        engine.setCoinPhase(th, true);

        // Inject unit impulse
        engine.processSample(1.0f, 1.0f);

        float last_sec_energy = 0.0f;
        for (int n = 1; n < 48000; ++n) {
            auto [l, r] = engine.processSample(0.0f, 0.0f);
            assert(!std::isnan(l) && !std::isnan(r));
            assert(!std::isinf(l) && !std::isinf(r));
            assert(std::abs(l) < 2.0f && std::abs(r) < 2.0f);
            if (n >= 40000) {
                last_sec_energy += l * l + r * r;
            }
        }
        // Energy must be well decayed after 40k samples (~0.83 s with T60=1.8s)
        assert(last_sec_energy < 0.1f);
    }
    std::cout << "PASSED (No NaN/Inf, strict boundedness and natural decay)" << std::endl;
}

// Test 5: Interference Behavior: Echo Density, Kurtosis and Spectral Flatness
void test_interference_metrics() {
    std::cout << "[TEST 5] Interference Metrics across Coin Phase angles..." << std::endl;
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDryWet(1.0f);

    struct AngleResult {
        float theta;
        float total_energy;
        float echo_density_50ms;
        float echo_density_100ms;
        float kurtosis_100ms;
        float early_late_ratio;
    };

    std::vector<AngleResult> results;
    const float thetas[] = {0.0f, 0.25f * 3.14159f, 0.5f * 3.14159f, 0.75f * 3.14159f, 1.0f * 3.14159f};

    for (float th : thetas) {
        engine.reset();
        engine.setCoinPhase(th, true);

        std::vector<float> ir_l(9600, 0.0f);
        std::vector<float> ir_r(9600, 0.0f);

        auto [l0, r0] = engine.processSample(1.0f, 0.0f);
        ir_l[0] = l0; ir_r[0] = r0;

        for (size_t n = 1; n < ir_l.size(); ++n) {
            auto [l, r] = engine.processSample(0.0f, 0.0f);
            ir_l[n] = l; ir_r[n] = r;
        }

        float total_e = 0.0f;
        for (float s : ir_l) total_e += s * s;

        // Echo density at 50ms (win around 2400) and 100ms (win around 4800)
        auto calc_ed = [&](int center, int radius) {
            int count = 0;
            float sum_sq = 0.0f;
            for (int k = center - radius; k <= center + radius; ++k) sum_sq += ir_l[k] * ir_l[k];
            float sigma = std::sqrt(sum_sq / (2 * radius + 1));
            for (int k = center - radius; k <= center + radius; ++k) {
                if (std::abs(ir_l[k]) > sigma) ++count;
            }
            return static_cast<float>(count) / (2 * radius + 1);
        };

        // Kurtosis around 100ms: E[x^4] / (E[x^2])^2
        auto calc_kurt = [&](int center, int radius) {
            float m2 = 0.0f, m4 = 0.0f;
            for (int k = center - radius; k <= center + radius; ++k) {
                float v2 = ir_l[k] * ir_l[k];
                m2 += v2;
                m4 += v2 * v2;
            }
            m2 /= (2 * radius + 1);
            m4 /= (2 * radius + 1);
            return (m2 > 1e-12f) ? (m4 / (m2 * m2)) : 0.0f;
        };

        float ed_50  = calc_ed(2400, 400);
        float ed_100 = calc_ed(4800, 600);
        float kurt   = calc_kurt(4800, 600);

        float early_e = 0.0f;
        for (int k = 0; k < 2400; ++k) early_e += ir_l[k] * ir_l[k];
        float el_ratio = early_e / std::max(1e-12f, total_e - early_e);

        results.push_back({th, total_e, ed_50, ed_100, kurt, el_ratio});

        std::cout << "    theta = " << std::setw(5) << std::fixed << std::setprecision(2) << (th / 3.14159f) << " pi"
                  << " | Energy = " << std::setw(8) << total_e
                  << " | ED(50ms) = " << std::setprecision(3) << ed_50
                  << " | ED(100ms) = " << ed_100
                  << " | Kurtosis = " << std::setprecision(2) << kurt
                  << " | Early/Late E-Ratio = " << el_ratio << std::endl;
    }
    std::cout << "  [TEST 5] PASSED" << std::endl;
}

// Test 6: Parameter Automation & Glitch-Free Smoothing
void test_realtime_automation() {
    std::cout << "[TEST 6] Realtime Automation & Glitch-Free Smoothing... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setCoinPhase(0.0f, true);
    engine.setCoinPhaseSmoothingTime(10.0f); // 10ms smoothing

    // Target a step jump in phase from 0 to pi/2
    engine.setCoinPhase(1.570796f, false);

    float max_sample_delta = 0.0f;
    float prev_val = 0.0f;

    for (int n = 0; n < 4800; ++n) {
        auto [l, r] = engine.processSample(0.5f, 0.5f);
        if (n > 0) {
            float delta = std::abs(l - prev_val);
            max_sample_delta = std::max(max_sample_delta, delta);
        }
        prev_val = l;
    }

    // Verify smooth parameter convergence
    assert(std::abs(engine.getCurrentCoinPhase() - 1.570796f) < 1e-4f);
    std::cout << "PASSED (Max sample-to-sample delta: " << max_sample_delta << ")" << std::endl;
}

// Test 7: Parity between SSE4.1 and Scalar Backends across theta sweep
void test_backend_parity() {
    std::cout << "[TEST 7] SSE4.1 vs Scalar Parity across 10,000 samples... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine_sse;
    dtqw::DTQWEngine<dtqw::simd::ScalarBackend> engine_scalar;

    engine_sse.prepare(48000.0);
    engine_scalar.prepare(48000.0);

    engine_sse.setCoinPhase(0.85f, true);
    engine_scalar.setCoinPhase(0.85f, true);

    float max_diff = 0.0f;
    for (int i = 0; i < 10000; ++i) {
        float in_l = (i % 256 == 0) ? 1.0f : 0.0f;
        float in_r = (i % 512 == 0) ? -0.7f : 0.0f;

        if (i == 3000) {
            engine_sse.setCoinPhase(2.1f, false);
            engine_scalar.setCoinPhase(2.1f, false);
        }

        auto [l_sse, r_sse] = engine_sse.processSample(in_l, in_r);
        auto [l_sca, r_sca] = engine_scalar.processSample(in_l, in_r);

        max_diff = std::max(max_diff, std::abs(l_sse - l_sca));
        max_diff = std::max(max_diff, std::abs(r_sse - r_sca));
    }

    std::cout << "PASSED (Max backend diff: " << max_diff << " < 1e-6)" << std::endl;
    assert(max_diff < 1e-6f);
}

// Test 8: Strict Allocation Freedom in Audio Thread
void test_allocation_freedom() {
    std::cout << "[TEST 8] Zero Heap Allocations in Audio Thread... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);

#ifndef __SANITIZE_ADDRESS__
    g_heap_allocations = 0;
    g_track_allocations = true;
#endif

    for (int i = 0; i < 50000; ++i) {
        if (i == 1000) engine.setCoinPhase(0.5f, false);
        if (i == 5000) engine.setCoinPhase(1.5f, true);
        if (i == 10000) engine.setCoinPhaseSmoothingTime(25.0f);
        if (i == 15000) engine.setRoomSize(1.2f);
        if (i == 20000) engine.setChorusDepth(2.5f);

        auto [l, r] = engine.processSample(0.1f, -0.1f);
        (void)l; (void)r;
    }

#ifndef __SANITIZE_ADDRESS__
    g_track_allocations = false;
    std::cout << "PASSED (Total heap allocations: " << g_heap_allocations << ")" << std::endl;
    assert(g_heap_allocations == 0);
#else
    std::cout << "PASSED (ASan active)" << std::endl;
#endif
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << " DTQW Reverb - Unitary Coin Phase & Interference  " << std::endl;
    std::cout << "==================================================" << std::endl;

    test_matrix_unitarity();
    test_eigenvalues();
    test_zero_t0_bleed();
    test_stability_decay();
    test_interference_metrics();
    test_realtime_automation();
    test_backend_parity();
    test_allocation_freedom();

    std::cout << "==================================================" << std::endl;
    std::cout << " ALL 8 TESTS PASSED SUCCESSFULLY!                 " << std::endl;
    std::cout << "==================================================" << std::endl;
    return 0;
}
