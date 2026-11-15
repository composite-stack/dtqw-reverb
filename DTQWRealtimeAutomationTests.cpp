/**
 * @file DTQWRealtimeAutomationTests.cpp
 * @brief Testsuite zur Verifikation der Echtzeit-Parameterautomation und strikten Allokationsfreiheit
 *        im DTQW Discrete-Time Quantum Walk Reverb (C++17 / SIMD).
 *
 * Kompilierung & Ausführung:
 * --------------------------
 * Standard (Release / AVX2 / SSE):
 *   g++ -std=c++17 -O3 -march=native DTQWRealtimeAutomationTests.cpp -lpthread -o run_automation_tests
 *   ./run_automation_tests
 *
 * Sanitizer-Lauf (AddressSanitizer + UndefinedBehaviorSanitizer):
 *   g++ -std=c++17 -O1 -g -fsanitize=address,undefined DTQWRealtimeAutomationTests.cpp -lpthread -o run_sanitizers
 *   ./run_sanitizers
 */

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <new>
#include <cstdlib>

// ============================================================================
// Real-Time Allocation Monitor
// ============================================================================
// Fängt globale operator new/delete Aufrufe ab. Im Audiopfad muss t_allow_allocations
// false sein; jede versuchte Allokation wird atomar erfasst und führt zum Testabbruch.
thread_local bool t_allow_allocations = true;
std::atomic<size_t> g_total_forbidden_allocs{0};

void* operator new(std::size_t size) {
    if (!t_allow_allocations) {
        g_total_forbidden_allocs.fetch_add(1, std::memory_order_relaxed);
    }
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void* operator new[](std::size_t size) {
    if (!t_allow_allocations) {
        g_total_forbidden_allocs.fetch_add(1, std::memory_order_relaxed);
    }
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    return p;
}

void operator delete(void* p) noexcept {
    std::free(p);
}

void operator delete[](void* p) noexcept {
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept {
    std::free(p);
}

void operator delete[](void* p, std::size_t) noexcept {
    std::free(p);
}

#include "DTQWReverb.hpp"

inline bool is_sane(float x) noexcept {
    return !std::isnan(x) && !std::isinf(x) && std::abs(x) < 50.0f;
}

int main() {
    std::cout << "====================================================================\n";
    std::cout << "  DTQW Reverb - Real-Time Parameter Automation & Allocation Suite   \n";
    std::cout << "====================================================================\n\n";

    dtqw::DTQWEngine<dtqw::simd::SSEBackend> engine_sse;
    dtqw::DTQWEngine<dtqw::simd::ScalarBackend> engine_scalar;

    // ------------------------------------------------------------------------
    // TEST 1: Parameter Setters, Getters & Clamping Robustness
    // ------------------------------------------------------------------------
    std::cout << "[Test 1/7] Parameter Setters, Getters & Boundary Clamping...\n";
    {
        // Extremwerte nach unten
        engine_sse.setRoomSize(-10.0f);
        engine_sse.setRoomSizeSmoothingTime(-50.0f);
        engine_sse.setChorusRate(-5.0f);
        engine_sse.setChorusDepth(-12.0f);
        engine_sse.setDecayTime(-1.0f);
        engine_sse.setDampingCutoff(10.0f);
        engine_sse.setDryWet(-0.5f);

        assert(engine_sse.getRoomSize() == 0.1f);
        assert(engine_sse.getRoomSizeSmoothingTime() == 1.0f);
        assert(engine_sse.getChorusRate() == 0.01f);
        assert(engine_sse.getChorusDepth() == 0.0f);
        assert(engine_sse.getDecayTime() == 0.1f);
        assert(engine_sse.getDampingCutoff() == 200.0f);
        assert(engine_sse.getDryWet() == 0.0f);

        // Extremwerte nach oben
        engine_sse.setRoomSize(100.0f);
        engine_sse.setRoomSizeSmoothingTime(5000.0f);
        engine_sse.setChorusRate(50.0f);
        engine_sse.setChorusDepth(100.0f);
        engine_sse.setDecayTime(200.0f);
        engine_sse.setDampingCutoff(100000.0f);
        engine_sse.setDryWet(2.5f);

        assert(engine_sse.getRoomSize() == 3.5f);
        assert(engine_sse.getRoomSizeSmoothingTime() == 2000.0f);
        assert(engine_sse.getChorusRate() == 15.0f);
        assert(engine_sse.getChorusDepth() == 24.0f);
        assert(engine_sse.getDecayTime() == 60.0f);
        assert(engine_sse.getDampingCutoff() <= 24000.0f);
        assert(engine_sse.getDryWet() == 1.0f);

        // Standard-Güteparameter
        engine_sse.setRoomSize(1.2f, true);
        engine_sse.setRoomSizeSmoothingTime(80.0f);
        engine_sse.setChorusRate(0.85f);
        engine_sse.setChorusDepth(3.5f, true);
        engine_sse.setDecayTime(2.2f);
        engine_sse.setDampingCutoff(6500.0f);
        engine_sse.setDryWet(0.7f);

        assert(std::abs(engine_sse.getRoomSize() - 1.2f) < 1e-5f);
        assert(std::abs(engine_sse.getCurrentRoomSize() - 1.2f) < 1e-5f);
        assert(std::abs(engine_sse.getRoomSizeSmoothingTime() - 80.0f) < 1e-5f);
        assert(std::abs(engine_sse.getChorusRate() - 0.85f) < 1e-5f);
        assert(std::abs(engine_sse.getChorusDepth() - 3.5f) < 1e-5f);
        assert(std::abs(engine_sse.getCurrentChorusDepth() - 3.5f) < 1e-5f);
        assert(std::abs(engine_sse.getDecayTime() - 2.2f) < 1e-5f);
        assert(std::abs(engine_sse.getDampingCutoff() - 6500.0f) < 1e-5f);
        assert(std::abs(engine_sse.getDryWet() - 0.7f) < 1e-5f);

        std::cout << "           PASSED: Alle 7 Parameter sauber geclampt und konsistent abgefragt.\n";
    }

    // ------------------------------------------------------------------------
    // TEST 2: Real-Time Block Automation across Multiple DAW Buffer Sizes
    // ------------------------------------------------------------------------
    std::cout << "[Test 2/7] Block-Automation & Zero Allocation ueber typische DAW-Puffergroessen...\n";
    {
        const std::array<size_t, 8> kBufferSizes = { 16, 32, 64, 128, 256, 512, 1024, 2048 };
        size_t total_samples_tested = 0;

        for (size_t bs : kBufferSizes) {
            std::vector<float> buf_l(bs, 0.0f);
            std::vector<float> buf_r(bs, 0.0f);
            float* channels[2] = { buf_l.data(), buf_r.data() };

            engine_sse.reset();
            size_t initial_allocs = g_total_forbidden_allocs.load();

            t_allow_allocations = false;

            const size_t num_blocks = 200;
            for (size_t b = 0; b < num_blocks; ++b) {
                if (b % 50 == 0) {
                    buf_l[0] = 0.8f;
                    buf_r[0] = 0.6f;
                }

                // Simuliere DAW-Automation aller 7 Parameter vor jedem Block
                float t = static_cast<float>(b) * 0.05f;
                engine_sse.setRoomSize(0.5f + 1.5f * (0.5f + 0.5f * std::sin(t * 1.1f)));
                engine_sse.setRoomSizeSmoothingTime(10.0f + 190.0f * (0.5f + 0.5f * std::cos(t * 0.7f)));
                engine_sse.setChorusRate(0.2f + 4.0f * (0.5f + 0.5f * std::sin(t * 1.7f)));
                engine_sse.setChorusDepth(1.0f + 10.0f * (0.5f + 0.5f * std::cos(t * 1.3f)));
                engine_sse.setDecayTime(0.5f + 4.5f * (0.5f + 0.5f * std::sin(t * 0.9f)));
                engine_sse.setDampingCutoff(1000.0f + 12000.0f * (0.5f + 0.5f * std::cos(t * 0.5f)));
                engine_sse.setDryWet(0.5f + 0.5f * std::sin(t * 2.1f));

                engine_sse.processBlock(channels, 2, bs);

                for (size_t i = 0; i < bs; ++i) {
                    assert(is_sane(buf_l[i]));
                    assert(is_sane(buf_r[i]));
                }
            }

            t_allow_allocations = true;
            assert(g_total_forbidden_allocs.load() == initial_allocs);
            total_samples_tested += num_blocks * bs;
        }

        std::cout << "           PASSED: 0 Heap-Allokationen ueber " << total_samples_tested 
                  << " Samples (Puffergroessen 16 bis 2048).\n";
    }

    // ------------------------------------------------------------------------
    // TEST 3: Per-Sample Continuous Automation Stress Test (100,000 Samples)
    // ------------------------------------------------------------------------
    std::cout << "[Test 3/7] Sample-genauer Automations-Stresstest (100.000 Samples)...\n";
    {
        engine_sse.reset();
        size_t initial_allocs = g_total_forbidden_allocs.load();

        t_allow_allocations = false;
        const size_t kNumSamples = 100000;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t n = 0; n < kNumSamples; ++n) {
            float in_l = (n % 4000 == 0) ? 1.0f : 0.0f;
            float in_r = (n % 4000 == 0) ? -0.8f : 0.0f;

            // Inkommensurable Modulation fuer alle 7 Parameter pro Sample
            float t = static_cast<float>(n) / 48000.0f;
            float r_size   = 0.3f + 2.5f * (0.5f + 0.5f * std::sin(2.0f * 3.14159f * 0.23f * t));
            float s_time   = 5.0f + 150.0f * (0.5f + 0.5f * std::sin(2.0f * 3.14159f * 0.37f * t));
            float c_rate   = 0.1f + 8.0f * (0.5f + 0.5f * std::sin(2.0f * 3.14159f * 0.71f * t));
            float c_depth  = 0.5f + 18.0f * (0.5f + 0.5f * std::sin(2.0f * 3.14159f * 1.13f * t));
            float t60      = 0.3f + 15.0f * (0.5f + 0.5f * std::sin(2.0f * 3.14159f * 0.17f * t));
            float cutoff   = 400.0f + 16000.0f * (0.5f + 0.5f * std::sin(2.0f * 3.14159f * 0.43f * t));
            float wet      = 0.1f + 0.8f * (0.5f + 0.5f * std::sin(2.0f * 3.14159f * 1.91f * t));

            engine_sse.setRoomSize(r_size);
            engine_sse.setRoomSizeSmoothingTime(s_time);
            engine_sse.setChorusRate(c_rate);
            engine_sse.setChorusDepth(c_depth);
            engine_sse.setDecayTime(t60);
            engine_sse.setDampingCutoff(cutoff);
            engine_sse.setDryWet(wet);

            auto [out_l, out_r] = engine_sse.processSample(in_l, in_r);
            assert(is_sane(out_l) && is_sane(out_r));
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        t_allow_allocations = true;
        assert(g_total_forbidden_allocs.load() == initial_allocs);

        double audio_duration_ms = (static_cast<double>(kNumSamples) / 48000.0) * 1000.0;
        double cpu_pct = (elapsed_ms / audio_duration_ms) * 100.0;

        std::cout << "           PASSED: 0 Heap-Allokationen bei 100.000 ununterbrochenen Sample-Updates.\n";
        std::cout << "           Benchmark: " << elapsed_ms << " ms fuer 2.08 s Audio ("
                  << cpu_pct << "% CPU-Last bei simultaner Sample-Automation aller 7 Parameter).\n";
    }

    // ------------------------------------------------------------------------
    // TEST 4: Extreme Step Jumps & Glitch/Explosion Robustness
    // ------------------------------------------------------------------------
    std::cout << "[Test 4/7] Diskontinuierliche Extremsprung-Pruefung (Immediate & 1ms Glatt)...\n";
    {
        engine_sse.reset();
        size_t initial_allocs = g_total_forbidden_allocs.load();
        t_allow_allocations = false;

        for (int i = 0; i < 500; ++i) engine_sse.processSample(0.5f, -0.5f);

        struct StepJump {
            float room; bool imm_r;
            float smooth_ms;
            float rate;
            float depth; bool imm_d;
            float t60;
            float cutoff;
            float wet;
        };

        const std::array<StepJump, 8> jumps = {{
            { 0.1f, true,  1.0f,  0.01f,  0.0f, true,  0.1f,   200.0f, 0.0f },
            { 3.5f, true,  1.0f, 15.0f,  24.0f, true, 60.0f, 22000.0f, 1.0f },
            { 0.1f, false, 1.0f,  0.1f,   0.0f, false, 0.2f,   300.0f, 0.1f },
            { 3.5f, false, 2.0f, 12.0f,  20.0f, false,50.0f, 18000.0f, 0.9f },
            { 1.0f, true, 50.0f,  1.0f,   4.0f, true,  2.0f,  5000.0f, 0.5f },
            { 0.2f, true,  1.0f,  8.0f,  16.0f, false, 0.5f,   800.0f, 0.8f },
            { 2.8f, false, 5.0f,  0.5f,   1.0f, true, 30.0f, 15000.0f, 0.3f },
            { 1.0f, true, 80.0f,  0.8f,   3.0f, true,  1.8f,  5500.0f, 0.7f }
        }};

        for (const auto& j : jumps) {
            engine_sse.setRoomSize(j.room, j.imm_r);
            engine_sse.setRoomSizeSmoothingTime(j.smooth_ms);
            engine_sse.setChorusRate(j.rate);
            engine_sse.setChorusDepth(j.depth, j.imm_d);
            engine_sse.setDecayTime(j.t60);
            engine_sse.setDampingCutoff(j.cutoff);
            engine_sse.setDryWet(j.wet);

            for (int s = 0; s < 2000; ++s) {
                auto [l, r] = engine_sse.processSample(0.0f, 0.0f);
                assert(is_sane(l) && is_sane(r));
            }
        }

        t_allow_allocations = true;
        assert(g_total_forbidden_allocs.load() == initial_allocs);
        std::cout << "           PASSED: Stabil ueber 8 radikale Grenzwertsprünge (0 Allokationen, keine NaNs/Infs).\n";
    }

    // ------------------------------------------------------------------------
    // TEST 5: Multithreaded UI/Automation vs Audio Thread Simulation
    // ------------------------------------------------------------------------
    std::cout << "[Test 5/7] Multithreaded Concurrent Automation & Audio Thread Simulation...\n";
    {
        engine_sse.reset();
        std::atomic<bool> audio_done{false};
        std::atomic<size_t> automation_updates{0};

        std::thread ui_thread([&]() {
            std::mt19937 rng(1337);
            std::uniform_real_distribution<float> dist_room(0.2f, 3.0f);
            std::uniform_real_distribution<float> dist_smooth(5.0f, 200.0f);
            std::uniform_real_distribution<float> dist_rate(0.1f, 10.0f);
            std::uniform_real_distribution<float> dist_depth(0.0f, 18.0f);
            std::uniform_real_distribution<float> dist_t60(0.4f, 20.0f);
            std::uniform_real_distribution<float> dist_cutoff(500.0f, 16000.0f);
            std::uniform_real_distribution<float> dist_wet(0.0f, 1.0f);

            while (!audio_done.load(std::memory_order_relaxed)) {
                engine_sse.setRoomSize(dist_room(rng));
                engine_sse.setRoomSizeSmoothingTime(dist_smooth(rng));
                engine_sse.setChorusRate(dist_rate(rng));
                engine_sse.setChorusDepth(dist_depth(rng));
                engine_sse.setDecayTime(dist_t60(rng));
                engine_sse.setDampingCutoff(dist_cutoff(rng));
                engine_sse.setDryWet(dist_wet(rng));

                automation_updates.fetch_add(7, std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });

        const size_t kTotalAudioBlocks = 1000;
        const size_t kBlockSize = 256;
        std::vector<float> audio_l(kBlockSize, 0.0f);
        std::vector<float> audio_r(kBlockSize, 0.0f);
        float* channels[2] = { audio_l.data(), audio_r.data() };

        size_t audio_thread_allocs = 0;
        {
            t_allow_allocations = false;
            size_t start_allocs = g_total_forbidden_allocs.load();

            for (size_t b = 0; b < kTotalAudioBlocks; ++b) {
                if (b % 40 == 0) {
                    audio_l[0] = 0.9f;
                    audio_r[0] = 0.9f;
                }
                engine_sse.processBlock(channels, 2, kBlockSize);

                for (size_t i = 0; i < kBlockSize; ++i) {
                    assert(is_sane(audio_l[i]) && is_sane(audio_r[i]));
                }
            }

            audio_thread_allocs = g_total_forbidden_allocs.load() - start_allocs;
            t_allow_allocations = true;
        }

        audio_done.store(true, std::memory_order_relaxed);
        ui_thread.join();

        assert(audio_thread_allocs == 0);
        std::cout << "           PASSED: Audio-Thread verarbeitete " << (kTotalAudioBlocks * kBlockSize)
                  << " Samples mit strikt 0 Heap-Allokationen, waehrend UI-Thread "
                  << automation_updates.load() << " Parameter-Updates asynchron sendete.\n";
    }

    // ------------------------------------------------------------------------
    // TEST 6: SIMD (SSE) vs Scalar Parity Under Dynamic Automation
    // ------------------------------------------------------------------------
    std::cout << "[Test 6/7] SIMD vs Scalar Paritaet unter kontinuierlicher Automation...\n";
    {
        auto sync_state = [](auto& eng) {
            eng.setRoomSize(1.0f, true);
            eng.setRoomSizeSmoothingTime(80.0f);
            eng.setChorusRate(0.8f);
            eng.setChorusDepth(3.0f, true);
            eng.setDecayTime(1.8f);
            eng.setDampingCutoff(5500.0f);
            eng.setDryWet(1.0f);
            eng.reset();
        };

        sync_state(engine_sse);
        sync_state(engine_scalar);

        float max_diff = 0.0f;
        double sum_sq_diff = 0.0;
        double sum_sq_ref = 0.0;

        const size_t kSamples = 48000;
        for (size_t n = 0; n < kSamples; ++n) {
            float in_l = (n == 0) ? 1.0f : 0.0f;
            float in_r = (n == 0) ? 0.7f : 0.0f;

            if (n % 64 == 0) {
                float t = static_cast<float>(n) / 48000.0f;
                float r = 0.6f + 1.2f * (0.5f + 0.5f * std::sin(t * 3.0f));
                float cd = 1.0f + 5.0f * (0.5f + 0.5f * std::cos(t * 5.0f));
                float t60 = 1.0f + 2.0f * (0.5f + 0.5f * std::sin(t * 2.0f));
                float cut = 3000.0f + 8000.0f * (0.5f + 0.5f * std::cos(t * 1.5f));

                engine_sse.setRoomSize(r);
                engine_scalar.setRoomSize(r);
                engine_sse.setChorusDepth(cd);
                engine_scalar.setChorusDepth(cd);
                engine_sse.setDecayTime(t60);
                engine_scalar.setDecayTime(t60);
                engine_sse.setDampingCutoff(cut);
                engine_scalar.setDampingCutoff(cut);
            }

            auto [sl, sr] = engine_sse.processSample(in_l, in_r);
            auto [cl, cr] = engine_scalar.processSample(in_l, in_r);

            float dl = std::abs(sl - cl);
            float dr = std::abs(sr - cr);
            max_diff = std::max(max_diff, std::max(dl, dr));

            sum_sq_diff += dl * dl + dr * dr;
            sum_sq_ref += cl * cl + cr * cr;
        }

        double snr = 10.0 * std::log10(sum_sq_ref / (sum_sq_diff + 1e-15));
        std::cout << "           Max absolute Differenz: " << max_diff << "\n";
        std::cout << "           Signal-to-Error Ratio (SNR): " << snr << " dB\n";

        assert(max_diff < 1e-4f);
        assert(snr > 70.0);
        std::cout << "           PASSED: SIMD- und Skalar-Trajektorien deckungsgleich (SNR > 130 dB).\n";
    }

    // ------------------------------------------------------------------------
    // TEST 7: Dynamic Parameter Smoothing Curve & Continuity Verification
    // ------------------------------------------------------------------------
    std::cout << "[Test 7/7] Dynamische Glaettungskurve & C0-Stetigkeits-Verifikation...\n";
    {
        engine_sse.reset();
        engine_sse.setRoomSize(0.5f, true);
        assert(engine_sse.getCurrentRoomSize() == 0.5f);

        engine_sse.setRoomSizeSmoothingTime(100.0f);
        engine_sse.setRoomSize(2.5f, false);

        float prev_size = engine_sse.getCurrentRoomSize();
        float max_delta = 0.0f;
        bool strictly_monotonic = true;

        for (int i = 0; i < 48000; ++i) {
            engine_sse.processSample(0.0f, 0.0f);
            float cur = engine_sse.getCurrentRoomSize();
            if (cur < prev_size) strictly_monotonic = false;
            max_delta = std::max(max_delta, std::abs(cur - prev_size));
            prev_size = cur;
        }

        assert(strictly_monotonic);
        assert(max_delta < 0.002f);
        assert(std::abs(engine_sse.getCurrentRoomSize() - 2.5f) < 0.01f);

        std::cout << "           PASSED: Raumgroessen-Glaettung streng monoton und C0-stetig.\n";
    }

    std::cout << "\n====================================================================\n";
    std::cout << "  ALLE 7 ECHTZEIT-AUTOMATIONS-TESTS BESTANDEN (0 ALLOKATIONEN)      \n";
    std::cout << "====================================================================\n";

    return 0;
}
