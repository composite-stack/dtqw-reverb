#include <iostream>
#include <vector>
#include <cmath>
#include <array>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <new>
#include <atomic>
#include <random>

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

// Utility: check subnormal / denormal
static inline bool is_subnormal(float val) {
    return std::fpclassify(val) == FP_SUBNORMAL;
}

// ============================================================================
// Test 1: Micro-Buffer Invariance & Prime-Length Blocks
// ============================================================================
void test_micro_buffer_invariance() {
    std::cout << "[TEST 1] Micro-Buffer & Prime-Length Block Invariance (1, 2, 7, 16, 32 vs Sample-by-Sample)... ";
    constexpr double kSampleRate = 48000.0;
    constexpr size_t kNumSamples = 30000;

    std::vector<float> in_l(kNumSamples), in_r(kNumSamples);
    for (size_t i = 0; i < kNumSamples; ++i) {
        float t = static_cast<float>(i) / 48000.0f;
        in_l[i] = 0.3f * std::sin(2.0f * static_cast<float>(M_PI) * 220.0f * t)
                + 0.15f * std::sin(2.0f * static_cast<float>(M_PI) * 880.0f * t);
        in_r[i] = 0.3f * std::cos(2.0f * static_cast<float>(M_PI) * 330.0f * t);
    }

    // Reference: sample-by-sample
    std::vector<float> ref_l(kNumSamples), ref_r(kNumSamples);
    {
        dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
        engine.prepare(kSampleRate);
        engine.setDispersion(0.65f, true);
        engine.setDecayTime(2.5f);
        engine.setChorusDepth(8.0f);
        engine.setRoomSize(1.2f, true);
        engine.setDryWet(1.0f);

        for (size_t i = 0; i < kNumSamples; ++i) {
            auto [l, r] = engine.processSample(in_l[i], in_r[i]);
            ref_l[i] = l;
            ref_r[i] = r;
        }
    }

    // Test block sizes: 1, 2, 7 (prime), 16, 31 (prime), 64
    const std::vector<size_t> test_block_sizes = { 1, 2, 7, 16, 31, 64 };

    for (size_t bs : test_block_sizes) {
        dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
        engine.prepare(kSampleRate);
        engine.setDispersion(0.65f, true);
        engine.setDecayTime(2.5f);
        engine.setChorusDepth(8.0f);
        engine.setRoomSize(1.2f, true);
        engine.setDryWet(1.0f);

        std::vector<float> out_l(kNumSamples), out_r(kNumSamples);
        float* channels[2];

        size_t processed = 0;
        while (processed < kNumSamples) {
            size_t current_block = std::min(bs, kNumSamples - processed);
            std::vector<float> buf_l(in_l.begin() + processed, in_l.begin() + processed + current_block);
            std::vector<float> buf_r(in_r.begin() + processed, in_r.begin() + processed + current_block);
            channels[0] = buf_l.data();
            channels[1] = buf_r.data();

            engine.processBlock(channels, 2, current_block);

            for (size_t j = 0; j < current_block; ++j) {
                out_l[processed + j] = buf_l[j];
                out_r[processed + j] = buf_r[j];
            }
            processed += current_block;
        }

        // Verify bit-exact equality with sample-by-sample
        for (size_t i = 0; i < kNumSamples; ++i) {
            assert(out_l[i] == ref_l[i]);
            assert(out_r[i] == ref_r[i]);
        }
    }

    std::cout << "PASSED (Bit-exact invariance across all micro/prime block sizes)\n";
}

// ============================================================================
// Test 2: Permanent High-Frequency Coefficient Automation & Decay Tail Stability
// ============================================================================
void test_continuous_coefficient_automation_and_tail_decay() {
    std::cout << "[TEST 2] Permanent High-Frequency Coefficient Automation & Tail Stability... ";
    constexpr double kSampleRate = 48000.0;
    constexpr size_t kActiveSamples = 48000;  // 1.0 s Input
    constexpr size_t kDecaySamples  = 96000;  // 2.0 s Decay
    constexpr size_t kTotalSamples  = kActiveSamples + kDecaySamples;

    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(kSampleRate);
    engine.setDispersion(0.0f, true);
    engine.setDecayTime(1.5f);
    engine.setDispersionCrossfadeTime(10.0f);
    engine.setDryWet(1.0f);

    float max_peak = 0.0f;
    size_t denormal_count = 0;

    // Small block processing: 16 samples per block
    constexpr size_t kBlockSize = 16;
    std::array<float, kBlockSize> block_l{}, block_r{};
    float* channels[2] = { block_l.data(), block_r.data() };

    for (size_t b = 0; b < kTotalSamples / kBlockSize; ++b) {
        size_t sample_offset = b * kBlockSize;

        // Permanent automation every single block: 25 Hz modulation of dispersion across [-0.90, +0.90]
        float mod_phase = 2.0f * static_cast<float>(M_PI) * 25.0f * (static_cast<float>(sample_offset) / 48000.0f);
        float target_g = 0.90f * std::sin(mod_phase);
        engine.setDispersion(target_g, false);

        // Feed input for first second, then silence to test decay tail
        for (size_t i = 0; i < kBlockSize; ++i) {
            if (sample_offset + i < kActiveSamples) {
                block_l[i] = 0.25f * std::sin(2.0f * static_cast<float>(M_PI) * 440.0f * static_cast<float>(sample_offset + i) / 48000.0f);
                block_r[i] = 0.25f * std::cos(2.0f * static_cast<float>(M_PI) * 550.0f * static_cast<float>(sample_offset + i) / 48000.0f);
            } else {
                block_l[i] = 0.0f;
                block_r[i] = 0.0f;
            }
        }

        engine.processBlock(channels, 2, kBlockSize);

        for (size_t i = 0; i < kBlockSize; ++i) {
            float vl = block_l[i];
            float vr = block_r[i];

            assert(!std::isnan(vl) && !std::isinf(vl));
            assert(!std::isnan(vr) && !std::isinf(vr));
            if (is_subnormal(vl) || is_subnormal(vr)) {
                ++denormal_count;
            }

            max_peak = std::max(max_peak, std::max(std::abs(vl), std::abs(vr)));
        }
    }

    // Verify stability: peak must remain reasonable (< 1.5)
    assert(max_peak < 1.5f);
    assert(denormal_count == 0);

    // Verify tail decay: after 2 seconds of silence, energy must be thoroughly decayed (< 1e-4)
    float tail_energy = 0.0f;
    for (size_t i = 0; i < kBlockSize; ++i) {
        tail_energy += block_l[i] * block_l[i] + block_r[i] * block_r[i];
    }
    assert(tail_energy < 1e-6f);

    std::cout << "PASSED (Max peak: " << max_peak << ", Denormals: " << denormal_count
              << ", Final tail energy: " << tail_energy << ")\n";
}

// ============================================================================
// Test 3: Micro-Retriggering Stress (Retrigger interval: 1, 3, 7, 13 samples)
// ============================================================================
void test_micro_retriggering_stress() {
    std::cout << "[TEST 3] Ultra-Dense Retriggering Stress (Intervals down to 1-4 Samples)... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDispersionCrossfadeTime(20.0f); // 20 ms Crossfade (960 Samples)
    engine.setDryWet(1.0f);

    constexpr size_t N = 40000;
    std::vector<float> out_l(N);

    // Feed a continuous bandlimited input
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist_g(-0.92f, +0.92f);
    std::uniform_int_distribution<size_t> dist_interval(1, 8); // Retrigger every 1 to 8 samples!

    size_t next_retrigger = 1;
    float max_sample_diff = 0.0f;
    float max_val = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        if (i == next_retrigger) {
            float g = dist_g(rng);
            engine.setDispersion(g, false);
            next_retrigger = i + dist_interval(rng);
        }

        float in = 0.2f * std::sin(2.0f * static_cast<float>(M_PI) * 350.0f * static_cast<float>(i) / 48000.0f);
        auto [l, r] = engine.processSample(in, in);

        assert(!std::isnan(l) && !std::isinf(l));
        out_l[i] = l;
        max_val = std::max(max_val, std::abs(l));

        if (i > 0) {
            max_sample_diff = std::max(max_sample_diff, std::abs(out_l[i] - out_l[i - 1]));
        }
    }

    // Max derivative step should remain completely smooth and bounded
    assert(max_val < 1.0f);
    assert(max_sample_diff < 0.05f);

    std::cout << "PASSED (Processed 40k samples with sub-8-sample retriggers, max diff: "
              << max_sample_diff << ", max peak: " << max_val << ")\n";
}

// ============================================================================
// Test 4: Extreme Polarity Alternation (+0.90 <-> -0.90 Square Wave Automation)
// ============================================================================
void test_extreme_polarity_square_wave_stress() {
    std::cout << "[TEST 4] Extreme Polarity Alternation (+0.90 <-> -0.90 Square Wave Retrigger)... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDispersionCrossfadeTime(15.0f); // 15 ms Crossfade (720 Samples)
    engine.setDryWet(1.0f);

    constexpr size_t N = 50000;
    // Retrigger every 32 samples (way before 720 samples finish!)
    constexpr size_t kRetriggerPeriod = 32;

    float max_peak = 0.0f;
    float max_deriv = 0.0f;
    float last_sample = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        if (i % kRetriggerPeriod == 0) {
            float g = ((i / kRetriggerPeriod) % 2 == 0) ? 0.90f : -0.90f;
            engine.setDispersion(g, false);
        }

        float in = 0.3f * std::sin(2.0f * static_cast<float>(M_PI) * 440.0f * static_cast<float>(i) / 48000.0f);
        auto [l, r] = engine.processSample(in, in);

        assert(!std::isnan(l) && !std::isinf(l));
        assert(!std::isnan(r) && !std::isinf(r));

        max_peak = std::max(max_peak, std::abs(l));
        if (i > 0) {
            max_deriv = std::max(max_deriv, std::abs(l - last_sample));
        }
        last_sample = l;
    }

    assert(max_peak < 1.5f);
    assert(max_deriv < 0.06f);

    std::cout << "PASSED (Max peak: " << max_peak << ", Max step derivative: " << max_deriv << ")\n";
}

// ============================================================================
// Test 5: Dynamic Buffer-Size Jitter with Concurrent Automation
// ============================================================================
void test_dynamic_buffer_size_jitter() {
    std::cout << "[TEST 5] Dynamic Buffer-Size Jitter (Random Blocks 1..48 Samples + Concurrent Retriggering)... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDispersionCrossfadeTime(12.0f);
    engine.setDryWet(1.0f);

    constexpr size_t N = 60000;
    std::vector<float> in_l(N), in_r(N);
    for (size_t i = 0; i < N; ++i) {
        in_l[i] = 0.2f * std::sin(2.0f * static_cast<float>(M_PI) * 200.0f * static_cast<float>(i) / 48000.0f);
        in_r[i] = 0.2f * std::cos(2.0f * static_cast<float>(M_PI) * 300.0f * static_cast<float>(i) / 48000.0f);
    }

    std::vector<float> out_l(N), out_r(N);
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist_block(1, 48); // Jittering buffer sizes between 1 and 48
    std::uniform_real_distribution<float> dist_disp(-0.85f, 0.85f);

    size_t processed = 0;
    size_t call_count = 0;

    while (processed < N) {
        size_t cur_block = std::min(dist_block(rng), N - processed);
        
        // Randomly automate dispersion on some block boundaries
        if (call_count % 3 == 0) {
            engine.setDispersion(dist_disp(rng), false);
        }

        std::vector<float> b_l(in_l.begin() + processed, in_l.begin() + processed + cur_block);
        std::vector<float> b_r(in_r.begin() + processed, in_r.begin() + processed + cur_block);
        float* channels[2] = { b_l.data(), b_r.data() };

        engine.processBlock(channels, 2, cur_block);

        for (size_t i = 0; i < cur_block; ++i) {
            assert(!std::isnan(b_l[i]) && !std::isinf(b_l[i]));
            assert(!std::isnan(b_r[i]) && !std::isinf(b_r[i]));
            out_l[processed + i] = b_l[i];
            out_r[processed + i] = b_r[i];
        }

        processed += cur_block;
        ++call_count;
    }

    std::cout << "PASSED (" << call_count << " buffer calls executed with jittered block sizes)\n";
}

// ============================================================================
// Test 6: Zero Heap Allocations under Combined Worst-Case Stress
// ============================================================================
void test_zero_allocations_combined_stress() {
    std::cout << "[TEST 6] Zero Heap Allocations under Combined Worst-Case Stress (100k Samples)... ";
    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine;
    engine.prepare(48000.0);
    engine.setDispersionCrossfadeTime(15.0f);

    g_audio_thread_allocations.store(0);
    g_track_allocations.store(true);

    constexpr size_t N = 100000;
    std::mt19937 rng(999);
    std::uniform_int_distribution<size_t> dist_block(1, 32);
    std::uniform_real_distribution<float> dist_g(-0.90f, 0.90f);

    // Pre-allocated stack buffer
    alignas(16) float buf_l[64];
    alignas(16) float buf_r[64];
    float* channels[2] = { buf_l, buf_r };

    size_t processed = 0;
    while (processed < N) {
        size_t cur_block = std::min(dist_block(rng), N - processed);

        if (processed % 64 == 0) {
            engine.setDispersion(dist_g(rng), false);
        }
        if (processed % 128 == 0) {
            engine.setPortDispersion({dist_g(rng), dist_g(rng), dist_g(rng), dist_g(rng)}, false);
        }

        for (size_t i = 0; i < cur_block; ++i) {
            buf_l[i] = 0.1f;
            buf_r[i] = 0.1f;
        }

        engine.processBlock(channels, 2, cur_block);
        processed += cur_block;
    }

    g_track_allocations.store(false);
    size_t allocs = g_audio_thread_allocations.load();
    assert(allocs == 0);

    std::cout << "PASSED (Strictly 0 heap allocations across 100,000 samples)\n";
}

// ============================================================================
// Test 7: SIMD Parity under Worst-Case Micro-Buffers and Rapid Retriggering
// ============================================================================
void test_simd_parity_worst_case() {
    std::cout << "[TEST 7] SIMD Parity under Worst-Case Stress (SSE vs NEON vs Scalar)... ";
    constexpr double kSampleRate = 48000.0;
    constexpr size_t kNumSamples = 20000;

    std::vector<float> input(kNumSamples);
    for (size_t i = 0; i < kNumSamples; ++i) {
        input[i] = 0.4f * std::sin(2.0f * static_cast<float>(M_PI) * 440.0f * static_cast<float>(i) / 48000.0f);
    }

    std::vector<float> out_scalar(kNumSamples), out_sse(kNumSamples), out_neon(kNumSamples);

    auto run_engine = [&](auto& engine, std::vector<float>& out) {
        engine.prepare(kSampleRate);
        engine.setDispersion(-0.7f, true);
        engine.setDispersionCrossfadeTime(15.0f);
        engine.setDryWet(1.0f);

        constexpr size_t kBlock = 16;
        float buf_l[kBlock], buf_r[kBlock];
        float* channels[2] = { buf_l, buf_r };

        for (size_t b = 0; b < kNumSamples / kBlock; ++b) {
            size_t offset = b * kBlock;
            if (b % 4 == 0) {
                float g = 0.85f * std::sin(static_cast<float>(b) * 0.3f);
                engine.setDispersion(g, false);
            }
            for (size_t i = 0; i < kBlock; ++i) {
                buf_l[i] = input[offset + i];
                buf_r[i] = input[offset + i];
            }
            engine.processBlock(channels, 2, kBlock);
            for (size_t i = 0; i < kBlock; ++i) {
                out[offset + i] = buf_l[i];
            }
        }
    };

    {
        dtqw::DTQWEngine<dtqw::simd::ScalarBackend> engine_scalar;
        run_engine(engine_scalar, out_scalar);
    }
    {
        dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine_sse;
        run_engine(engine_sse, out_sse);
    }
    {
        dtqw::DTQWEngine<dtqw::simd::NEONBackend> engine_neon;
        run_engine(engine_neon, out_neon);
    }

    double diff_sse = 0.0, diff_neon = 0.0, power_ref = 0.0;
    for (size_t i = 0; i < kNumSamples; ++i) {
        double d_sse = static_cast<double>(out_scalar[i] - out_sse[i]);
        double d_neon = static_cast<double>(out_scalar[i] - out_neon[i]);
        diff_sse += d_sse * d_sse;
        diff_neon += d_neon * d_neon;
        power_ref += static_cast<double>(out_scalar[i] * out_scalar[i]);
    }

    float snr_sse = static_cast<float>(10.0 * std::log10(power_ref / std::max(1e-15, diff_sse)));
    float snr_neon = static_cast<float>(10.0 * std::log10(power_ref / std::max(1e-15, diff_neon)));

    assert(snr_sse > 120.0f);
    assert(snr_neon > 120.0f);

    std::cout << "PASSED (SSE SNR: " << snr_sse << " dB, NEON SNR: " << snr_neon << " dB)\n";
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "    DTQW Reverb - Worst-Case & Extreme Stress Test Suite         \n";
    std::cout << "=================================================================\n";

    test_micro_buffer_invariance();
    test_continuous_coefficient_automation_and_tail_decay();
    test_micro_retriggering_stress();
    test_extreme_polarity_square_wave_stress();
    test_dynamic_buffer_size_jitter();
    test_zero_allocations_combined_stress();
    test_simd_parity_worst_case();

    std::cout << "=================================================================\n";
    std::cout << " SUMMARY: ALL 7 WORST-CASE STRESS TESTS PASSED SUCCESSFULLY!     \n";
    std::cout << "=================================================================\n";
    return 0;
}
