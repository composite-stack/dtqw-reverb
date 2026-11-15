#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>

#include "DTQWReverb.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct ScenarioMetric {
    std::string name;
    std::string backend;
    double sample_rate{48000.0};
    size_t block_size{128};
    double median_ms_per_sec{0.0};
    double min_ms_per_sec{0.0};
    double max_ms_per_sec{0.0};
    double stddev_ms_per_sec{0.0};
    double cpu_usage_pct{0.0};
    double realtime_factor{0.0};
};

struct BenchmarkConfig {
    std::string json_output_path = "";
    std::string baseline_path = "";
    double tolerance_pct = 25.0; // 25% Abweichungstoleranz für CI-Runner
    bool quiet = false;
    int warmup_runs = 3;
    int measure_runs = 11;
    double audio_duration_sec = 2.0;
};

template <typename Backend>
ScenarioMetric run_scenario(
    const std::string& name,
    const std::string& backend_name,
    double sample_rate,
    size_t block_size,
    bool dynamic_automation,
    const BenchmarkConfig& config)
{
    const size_t total_samples = static_cast<size_t>(sample_rate * config.audio_duration_sec);
    const size_t num_blocks = total_samples / block_size;
    const size_t actual_samples = num_blocks * block_size;
    const double actual_duration = static_cast<double>(actual_samples) / sample_rate;

    std::vector<float> in_l(block_size), in_r(block_size);
    for (size_t i = 0; i < block_size; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        in_l[i] = 0.25f * std::sin(2.0f * static_cast<float>(M_PI) * 440.0f * t);
        in_r[i] = 0.25f * std::cos(2.0f * static_cast<float>(M_PI) * 554.37f * t);
    }

    std::vector<float> buf_l(block_size), buf_r(block_size);
    float* channels[2] = { buf_l.data(), buf_r.data() };

    std::vector<double> run_times_ms;
    run_times_ms.reserve(config.measure_runs);

    for (int r = 0; r < config.warmup_runs + config.measure_runs; ++r) {
        dtqw::DTQWEngine<Backend, 2> engine;
        engine.prepare(sample_rate);
        engine.setRoomSize(1.2f);
        engine.setDecayTime(2.2f);
        engine.setDampingCutoff(6500.0f);
        engine.setChorusDepth(8.0f);
        engine.setChorusRate(0.8f);
        engine.setCoinPhase(0.35f);
        engine.setDispersion(0.45f, true);
        engine.setDispersionCrossfadeEnabled(true);
        engine.setDispersionCrossfadeTime(20.0f);
        engine.setDryWet(1.0f);

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t b = 0; b < num_blocks; ++b) {
            std::memcpy(buf_l.data(), in_l.data(), block_size * sizeof(float));
            std::memcpy(buf_r.data(), in_r.data(), block_size * sizeof(float));

            if (dynamic_automation) {
                // Dynamische Stress-Automation: Wechselnde Dispersion (Dual-Bank Crossfade) und Phasenwinkel
                float g = (b % 2 == 0) ? 0.70f : -0.70f;
                engine.setDispersion(g, false);
                engine.setCoinPhase((b % 4) * 0.25f);
                engine.setRoomSize(1.0f + 0.1f * std::sin(static_cast<float>(b) * 0.05f));
            }

            engine.processBlock(channels, 2, block_size);
        }

        auto end = std::chrono::high_resolution_clock::now();

        if (r >= config.warmup_runs) {
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            // Normalisiere auf Zeit (ms) pro 1.0 Sekunde Stereo-Audio
            double ms_per_sec = elapsed_ms / actual_duration;
            run_times_ms.push_back(ms_per_sec);
        }
    }

    std::sort(run_times_ms.begin(), run_times_ms.end());
    double median = run_times_ms[run_times_ms.size() / 2];
    double min_val = run_times_ms.front();
    double max_val = run_times_ms.back();

    double mean = std::accumulate(run_times_ms.begin(), run_times_ms.end(), 0.0) / run_times_ms.size();
    double sq_sum = 0.0;
    for (double v : run_times_ms) {
        sq_sum += (v - mean) * (v - mean);
    }
    double stddev = std::sqrt(sq_sum / run_times_ms.size());

    ScenarioMetric metric;
    metric.name = name;
    metric.backend = backend_name;
    metric.sample_rate = sample_rate;
    metric.block_size = block_size;
    metric.median_ms_per_sec = median;
    metric.min_ms_per_sec = min_val;
    metric.max_ms_per_sec = max_val;
    metric.stddev_ms_per_sec = stddev;
    metric.cpu_usage_pct = (median / 1000.0) * 100.0;
    metric.realtime_factor = 1000.0 / median;

    return metric;
}

void print_help() {
    std::cout << "DTQW Performance Benchmark\n"
              << "Optionen:\n"
              << "  --json <file>       Ergebnisse als JSON exportieren\n"
              << "  --baseline <file>   Gegen gespeicherte JSON-Baseline prüfen\n"
              << "  --tolerance <pct>   Maximal erlaubte prozentuale Abweichung (Standard: 25.0)\n"
              << "  --runs <n>          Anzahl Messläufe (Standard: 11)\n"
              << "  --warmup <n>        Anzahl Warmup-Läufe (Standard: 3)\n"
              << "  --duration <s>      Audiodauer pro Lauf in Sekunden (Standard: 2.0)\n"
              << "  --quiet             Kompakte Ausgabe\n"
              << "  --help              Diese Hilfe anzeigen\n";
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json" && i + 1 < argc) {
            config.json_output_path = argv[++i];
        } else if (arg == "--baseline" && i + 1 < argc) {
            config.baseline_path = argv[++i];
        } else if (arg == "--tolerance" && i + 1 < argc) {
            config.tolerance_pct = std::stod(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            config.measure_runs = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            config.warmup_runs = std::stoi(argv[++i]);
        } else if (arg == "--duration" && i + 1 < argc) {
            config.audio_duration_sec = std::stod(argv[++i]);
        } else if (arg == "--quiet") {
            config.quiet = true;
        } else if (arg == "--help") {
            print_help();
            return 0;
        }
    }

    if (!config.quiet) {
        std::cout << "========================================================================================\n"
                  << "  DTQW Reverb - Reproduzierbarer DSP Performance Benchmark\n"
                  << "========================================================================================\n";
    }

    std::vector<ScenarioMetric> results;

    // Repräsentative Szenarien:
    // 1. Standard Production: 48 kHz, Block 128, Steady-State (Allpass aktiv, LFOs aktiv)
    results.push_back(run_scenario<dtqw::simd::DefaultBackend>(
        "steady_state_48k_b128", "SIMD (Default)", 48000.0, 128, false, config));

    // 2. Low-Latency Studio: 48 kHz, Block 64, Steady-State
    results.push_back(run_scenario<dtqw::simd::DefaultBackend>(
        "low_latency_48k_b64", "SIMD (Default)", 48000.0, 64, false, config));

    // 3. Dynamic Automation / Dual-Bank Crossfade Stress: 48 kHz, Block 128
    results.push_back(run_scenario<dtqw::simd::DefaultBackend>(
        "dynamic_stress_48k_b128", "SIMD (Default)", 48000.0, 128, true, config));

    // 4. High-Resolution Audio: 96 kHz, Block 128, Steady-State
    results.push_back(run_scenario<dtqw::simd::DefaultBackend>(
        "hi_res_96k_b128", "SIMD (Default)", 96000.0, 128, false, config));

    // 5. Skalar-Referenz (zum Vergleich der SIMD-Effizienz): 48 kHz, Block 128
    results.push_back(run_scenario<dtqw::simd::ScalarBackend>(
        "scalar_ref_48k_b128", "Scalar", 48000.0, 128, false, config));

    if (!config.quiet) {
        std::cout << std::left
                  << std::setw(26) << "Szenario"
                  << std::setw(15) << "Backend"
                  << std::setw(8)  << "SR (Hz)"
                  << std::setw(7)  << "Block"
                  << std::setw(13) << "Zeit (ms/s)"
                  << std::setw(11) << "CPU %"
                  << std::setw(12) << "RTF"
                  << "\n";
        std::cout << "----------------------------------------------------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(2);

        for (const auto& m : results) {
            std::cout << std::left
                      << std::setw(26) << m.name
                      << std::setw(15) << m.backend
                      << std::setw(8)  << static_cast<int>(m.sample_rate)
                      << std::setw(7)  << m.block_size
                      << std::setw(13) << m.median_ms_per_sec
                      << std::setw(11) << m.cpu_usage_pct
                      << std::setw(12) << (std::to_string(static_cast<int>(m.realtime_factor)) + "x")
                      << "\n";
        }
        std::cout << "----------------------------------------------------------------------------------------\n";
    }

    // JSON Export wenn gewünscht
    if (!config.json_output_path.empty()) {
        std::ofstream jf(config.json_output_path);
        if (jf.is_open()) {
            jf << "{\n  \"scenarios\": {\n";
            for (size_t i = 0; i < results.size(); ++i) {
                const auto& m = results[i];
                jf << "    \"" << m.name << "\": {\n"
                   << "      \"backend\": \"" << m.backend << "\",\n"
                   << "      \"sample_rate\": " << m.sample_rate << ",\n"
                   << "      \"block_size\": " << m.block_size << ",\n"
                   << "      \"median_ms_per_sec\": " << m.median_ms_per_sec << ",\n"
                   << "      \"min_ms_per_sec\": " << m.min_ms_per_sec << ",\n"
                   << "      \"max_ms_per_sec\": " << m.max_ms_per_sec << ",\n"
                   << "      \"stddev_ms_per_sec\": " << m.stddev_ms_per_sec << ",\n"
                   << "      \"cpu_usage_pct\": " << m.cpu_usage_pct << ",\n"
                   << "      \"realtime_factor\": " << m.realtime_factor << "\n"
                   << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
            }
            jf << "  }\n}\n";
            if (!config.quiet) {
                std::cout << "Ergebnisse erfolgreich exportiert nach: " << config.json_output_path << "\n";
            }
        }
    }

    return 0;
}
