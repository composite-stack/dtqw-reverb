#include "WavFile.hpp"
#include "DTQWReverb.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdlib>

struct RenderConfig {
    std::string input_path;
    std::string output_path;

    bool is_ir_mode = false;
    double ir_duration = 3.0;
    double sample_rate = 48000.0;

    float room_size = 1.0f;
    float decay_t60 = 1.8f;
    float damping_cutoff = 5500.0f;
    float dry_wet = -1.0f; // -1 = auto (1.0 for IR, 0.35 for file)

    float coin_phase_rad = 0.0f;

    bool dispersion_enabled = false;
    float dispersion_g = 0.0f;
    float dispersion_freq = 0.0f; // 0 = use g directly
    float dispersion_crossfade_ms = 15.0f;

    float chorus_rate_hz = 0.8f;
    float chorus_depth_samples = 3.0f;

    int hadamard_in_l = 1;
    int hadamard_in_r = 2;
    int hadamard_out_l = 5;
    int hadamard_out_r = 6;

    double tail_seconds = -1.0; // -1 = auto based on T60

    dtqw::WavBitDepth bit_depth = dtqw::WavBitDepth::Bit24;
    float gain_db = 0.0f;
    bool normalize = false;
    float normalize_target_dbfs = -0.5f;

    size_t block_size = 256;
    bool quiet = false;
};

void printHelp(const char* prog_name) {
    std::cout << R"(
===============================================================================
  DTQW REVERB - OFFLINE WAV RENDER HARNESS (dtqw_render_wav)
  Discrete-Time Quantum Walk 4D-Hypercube Reverb Engine
===============================================================================

VERWENDUNG:
  # 1. Audiodatei mit Reverb verarbeiten:
  )" << prog_name << R"( [OPTIONEN] <input.wav> <output.wav>

  # 2. Impulsantwort (IR) synthetisieren:
  )" << prog_name << R"( [OPTIONEN] --ir <output.wav>

CORE REVERB PARAMETER:
  --room-size <val>          Raumgrößen-Skalierung [0.1 .. 3.5] (Default: 1.0)
  --decay <sek> | -t60 <sek> Nachhallzeit T60 in Sekunden [0.1 .. 60.0] (Default: 1.8)
  --damping <hz> | --cutoff  Dämpfungs-Grenzfrequenz in Hz [200 .. Nyquist] (Default: 5500)
  --mix <val> | --dry-wet    Dry/Wet Mischungsverhältnis [0.0 .. 1.0]
                             (Default: 1.0 für IR, 0.35 für Audio-Dateien)
  --coin-phase <rad>         Unitäre Coin-Phasenrotation in Radians [0.0 .. 2*pi) (Default: 0.0)
  --coin-phase-deg <grad>    Coin-Phasenrotation in Grad [0 .. 360] (z.B. 45, 90, 180)

DISPERSION (ALLPASS-KASKADEN IN DEN KANTEN):
  --dispersion <g> | -g <g>  Dispersions-Allpasskoeffizient g [-0.95 .. +0.95]
                             (> 0: Up-Chirp / Saitensteifigkeit, < 0: Down-Chirp)
  --dispersion-freq <hz>     Berechnet g über bilineare Transformation (-pi/2 bei freq)
  --dispersion-crossfade <ms> Dauer des zustandsübergreifenden Crossfades (Default: 15 ms)
  --no-dispersion            Schaltet die Allpass-Kaskaden explizit aus (Default bei g=0)

CHORUS / LFO MODULATION:
  --chorus-rate <hz>         Modulationsfrequenz in Hz [0.01 .. 15.0] (Default: 0.8)
  --chorus-depth <samples>   Modulationstiefe in Samples [0.0 .. 24.0] (Default: 3.0)
  --no-chorus                Deaktiviert Chorus-Modulation (Tiefe = 0.0)

SPATIAL HADAMARD MATRIXING:
  --hadamard-in <l> <r>      Injektions-Zeilen der 16x16 Walsh-Matrix [0 .. 15] (Default: 1 2)
  --hadamard-out <l> <r>     Auskopplungs-Zeilen der Walsh-Matrix [0 .. 15] (Default: 5 6)

RENDERING & AUSGABE:
  --ir                       Impulsantwort-Modus aktivieren (Dirac-Impuls als Eingang)
  --duration <sek>           Dauer der generierten IR in Sekunden (Default: 3.0)
  --sample-rate <hz>         Abtastrate für IR-Generierung in Hz (Default: 48000)
  --tail <sek>               Ausklingzeit (Tail) nach Dateiende in Sekunden
                             (Default: automatisch berechnet aus T60)
  --bit-depth <16|24|32>     Bit-Tiefe der Ausgabe-WAV (Default: 24)
  --float                    Abkürzung für 32-bit IEEE Float Ausgabe
  --gain-db <db>             Ausgangspegel-Trim in dB (z.B. -3.0 oder +2.5)
  --normalize [target_db]    Normalisiert das Ausgangssignal auf Ziel-dBFS (Default: -0.5 dBFS)
  --block-size <n>           Interne DSP-Blockgröße in Samples (Default: 256)
  --quiet | -q               Unterdrückt Fortschritts- und Statusmeldungen
  --help | -h                Zeigt diesen Hilfetext an

BEISPIELE:
  # Konzertsaal-Hall mit 2.5s Decay auf Gesang anwenden:
  dtqw_render_wav vocal.wav vocal_reverb.wav --decay 2.5 --room-size 1.2 --mix 0.25

  # Metallische Dispersion (g=0.6) und 45° Coin-Phasendrehung für Sounddesign:
  dtqw_render_wav snare.wav snare_dtqw.wav -g 0.6 --coin-phase-deg 45 --decay 3.0

  # Reine 4-Sekunden Stereo-Impulsantwort (32-bit Float) erzeugen:
  dtqw_render_wav --ir dtqw_space_ir.wav --duration 4.0 --float --decay 2.8 -g 0.4
===============================================================================
)";
}

bool parseCommandLine(int argc, char* argv[], RenderConfig& cfg) {
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printHelp(argv[0]);
            std::exit(0);
        } else if (arg == "--ir") {
            cfg.is_ir_mode = true;
        } else if (arg == "--room-size" && i + 1 < argc) {
            cfg.room_size = std::stof(argv[++i]);
        } else if ((arg == "--decay" || arg == "-t60") && i + 1 < argc) {
            cfg.decay_t60 = std::stof(argv[++i]);
        } else if ((arg == "--damping" || arg == "--cutoff") && i + 1 < argc) {
            cfg.damping_cutoff = std::stof(argv[++i]);
        } else if ((arg == "--mix" || arg == "--dry-wet") && i + 1 < argc) {
            cfg.dry_wet = std::stof(argv[++i]);
        } else if (arg == "--coin-phase" && i + 1 < argc) {
            cfg.coin_phase_rad = std::stof(argv[++i]);
        } else if (arg == "--coin-phase-deg" && i + 1 < argc) {
            float deg = std::stof(argv[++i]);
            cfg.coin_phase_rad = deg * static_cast<float>(M_PI / 180.0);
        } else if ((arg == "--dispersion" || arg == "-g") && i + 1 < argc) {
            cfg.dispersion_g = std::stof(argv[++i]);
            cfg.dispersion_enabled = (std::abs(cfg.dispersion_g) > 1e-5f);
        } else if (arg == "--dispersion-freq" && i + 1 < argc) {
            cfg.dispersion_freq = std::stof(argv[++i]);
            cfg.dispersion_enabled = true;
        } else if (arg == "--dispersion-crossfade" && i + 1 < argc) {
            cfg.dispersion_crossfade_ms = std::stof(argv[++i]);
        } else if (arg == "--no-dispersion") {
            cfg.dispersion_enabled = false;
            cfg.dispersion_g = 0.0f;
        } else if (arg == "--chorus-rate" && i + 1 < argc) {
            cfg.chorus_rate_hz = std::stof(argv[++i]);
        } else if (arg == "--chorus-depth" && i + 1 < argc) {
            cfg.chorus_depth_samples = std::stof(argv[++i]);
        } else if (arg == "--no-chorus") {
            cfg.chorus_depth_samples = 0.0f;
        } else if (arg == "--hadamard-in" && i + 2 < argc) {
            cfg.hadamard_in_l = std::stoi(argv[++i]);
            cfg.hadamard_in_r = std::stoi(argv[++i]);
        } else if (arg == "--hadamard-out" && i + 2 < argc) {
            cfg.hadamard_out_l = std::stoi(argv[++i]);
            cfg.hadamard_out_r = std::stoi(argv[++i]);
        } else if (arg == "--duration" && i + 1 < argc) {
            cfg.ir_duration = std::stod(argv[++i]);
            cfg.is_ir_mode = true;
        } else if (arg == "--sample-rate" && i + 1 < argc) {
            cfg.sample_rate = std::stod(argv[++i]);
        } else if (arg == "--tail" && i + 1 < argc) {
            cfg.tail_seconds = std::stod(argv[++i]);
        } else if (arg == "--bit-depth" && i + 1 < argc) {
            int bd = std::stoi(argv[++i]);
            if (bd == 16) cfg.bit_depth = dtqw::WavBitDepth::Bit16;
            else if (bd == 24) cfg.bit_depth = dtqw::WavBitDepth::Bit24;
            else if (bd == 32) cfg.bit_depth = dtqw::WavBitDepth::Float32;
            else {
                std::cerr << "Warnung: Ungültige Bit-Tiefe " << bd << ", verwende 24-bit.\n";
            }
        } else if (arg == "--float") {
            cfg.bit_depth = dtqw::WavBitDepth::Float32;
        } else if (arg == "--gain-db" && i + 1 < argc) {
            cfg.gain_db = std::stof(argv[++i]);
        } else if (arg == "--normalize") {
            cfg.normalize = true;
            if (i + 1 < argc) {
                const std::string next = argv[i + 1];
                bool numeric = !next.empty();
                size_t start = (next[0] == '-' || next[0] == '+') ? 1 : 0;
                bool has_digit = false;
                for (size_t j = start; j < next.size(); ++j) {
                    if (std::isdigit(static_cast<unsigned char>(next[j]))) has_digit = true;
                    else if (next[j] != '.') numeric = false;
                }
                numeric = numeric && has_digit;
                if (numeric) cfg.normalize_target_dbfs = std::stof(argv[++i]);
            }
        } else if (arg == "--block-size" && i + 1 < argc) {
            cfg.block_size = std::stoul(argv[++i]);
        } else if (arg == "--quiet" || arg == "-q") {
            cfg.quiet = true;
        } else if (arg[0] == '-' && (arg.size() < 2 || (!std::isdigit(static_cast<unsigned char>(arg[1])) && arg[1] != '.'))) {
            std::cerr << "Fehler: Unbekannte Option '" << arg << "'. Verwende --help für Hilfe.\n";
            return false;
        } else {
            positional.push_back(arg);
        }
    }

    if (cfg.is_ir_mode) {
        if (positional.empty()) {
            std::cerr << "Fehler: Kein Ausgabe-Dateipfad für IR angegeben. Verwende: " << argv[0] << " --ir <output.wav>\n";
            return false;
        }
        cfg.output_path = positional[0];
    } else {
        if (positional.size() == 1) {
            // Assume single file is IR output if flag was omitted or user intended IR
            std::cerr << "Hinweis: Nur eine Datei angegeben, interpretiere als IR-Ausgabe: " << positional[0] << "\n";
            cfg.is_ir_mode = true;
            cfg.output_path = positional[0];
        } else if (positional.size() >= 2) {
            cfg.input_path = positional[0];
            cfg.output_path = positional[1];
            if (positional.size() > 2) {
                std::cerr << "Fehler: Zu viele Positionsargumente. Verwende: <input.wav> <output.wav> [Optionen].\n";
                return false;
            }
        } else {
            printHelp(argv[0]);
            return false;
        }
    }

    if (cfg.dry_wet < 0.0f) {
        cfg.dry_wet = cfg.is_ir_mode ? 1.0f : 0.35f;
    }

    if (cfg.tail_seconds < 0.0) {
        cfg.tail_seconds = std::min<double>(cfg.decay_t60 * 1.2, 8.0);
    }

    if (!(cfg.sample_rate >= 8000.0 && cfg.sample_rate <= 192000.0)) {
        std::cerr << "Fehler: Sample-Rate muss zwischen 8000 und 192000 Hz liegen.\n";
        return false;
    }
    if (!(cfg.ir_duration > 0.0 && cfg.ir_duration <= 600.0)) {
        std::cerr << "Fehler: IR-Dauer muss groesser 0 und maximal 600 Sekunden sein.\n";
        return false;
    }
    if (!std::isfinite(cfg.room_size) || !(cfg.room_size >= 0.1f && cfg.room_size <= 3.5f)) {
        std::cerr << "Fehler: Raumgroesse muss im Bereich 0.1 bis 3.5 liegen.\n";
        return false;
    }
    if (!std::isfinite(cfg.decay_t60) || !(cfg.decay_t60 >= 0.1f && cfg.decay_t60 <= 60.0f)) {
        std::cerr << "Fehler: Decay muss zwischen 0.1 und 60 Sekunden liegen.\n";
        return false;
    }
    if (!std::isfinite(cfg.damping_cutoff) || cfg.damping_cutoff < 200.0f) {
        std::cerr << "Fehler: Daempfungs-Cutoff muss mindestens 200 Hz betragen.\n";
        return false;
    }
    if (!std::isfinite(cfg.dispersion_g) || cfg.dispersion_g < -0.95f || cfg.dispersion_g > 0.95f) {
        std::cerr << "Fehler: Dispersion g muss im Bereich -0.95 bis +0.95 liegen.\n";
        return false;
    }
    if (!std::isfinite(cfg.chorus_rate_hz) || !std::isfinite(cfg.chorus_depth_samples) ||
        cfg.chorus_rate_hz < 0.01f || cfg.chorus_rate_hz > 15.0f ||
        cfg.chorus_depth_samples < 0.0f || cfg.chorus_depth_samples > 24.0f) {
        std::cerr << "Fehler: Chorus-Rate/Tiefe ausserhalb des erlaubten Bereichs.\n";
        return false;
    }
    if (!std::isfinite(cfg.tail_seconds) || cfg.tail_seconds < 0.0 || cfg.tail_seconds > 600.0) {
        std::cerr << "Fehler: Tail muss zwischen 0 und 600 Sekunden liegen.\n";
        return false;
    }
    if (!std::isfinite(cfg.normalize_target_dbfs) || cfg.normalize_target_dbfs > 0.0f) {
        std::cerr << "Fehler: Normalisierungsziel darf 0 dBFS nicht ueberschreiten.\n";
        return false;
    }
    if (!std::isfinite(cfg.dry_wet) || !(cfg.dry_wet >= 0.0f && cfg.dry_wet <= 1.0f)) {
        std::cerr << "Fehler: Dry/Wet muss im Bereich 0 bis 1 liegen.\n";
        return false;
    }
    if (!(cfg.block_size >= 1 && cfg.block_size <= 65536)) {
        std::cerr << "Fehler: Blockgroesse muss zwischen 1 und 65536 Samples liegen.\n";
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    RenderConfig cfg;
    if (!parseCommandLine(argc, argv, cfg)) {
        return 1;
    }

    if (!cfg.quiet) {
        std::cout << "\n===============================================================================\n"
                  << "  DTQW REVERB - OFFLINE AUDIO RENDER ENGINE\n"
                  << "===============================================================================\n";
    }

    dtqw::WavReader reader;
    double sample_rate = cfg.sample_rate;
    size_t input_frames = 0;

    if (!cfg.is_ir_mode) {
        if (!reader.open(cfg.input_path)) {
            std::cerr << "Fehler beim Öffnen der Eingabedatei: " << reader.getError() << "\n";
            return 1;
        }
        sample_rate = reader.getSampleRate();
        input_frames = reader.getTotalFrames();

        if (!cfg.quiet) {
            std::cout << "EINGABE-DATEI:\n"
                      << "  Pfad:          " << cfg.input_path << "\n"
                      << "  Abtastrate:    " << reader.getSampleRate() << " Hz\n"
                      << "  Kanäle:        " << (reader.getChannels() == 1 ? "Mono (1)" : "Stereo (2)") << "\n"
                      << "  Bit-Tiefe:     " << reader.getBitsPerSample() << "-bit\n"
                      << "  Dauer:         " << std::fixed << std::setprecision(2) << reader.getDurationSeconds() << " s (" << input_frames << " Samples)\n\n";
        }
    } else {
        input_frames = 1;
        if (!cfg.quiet) {
            std::cout << "MODUS: IMPULSANNTWORT (IR SYNTHESE)\n"
                      << "  Abtastrate:    " << static_cast<uint32_t>(sample_rate) << " Hz\n"
                      << "  Dauer:         " << std::fixed << std::setprecision(2) << cfg.ir_duration << " s\n\n";
        }
    }

    if (!std::isfinite(cfg.dispersion_crossfade_ms) || cfg.dispersion_crossfade_ms < 1.0f || cfg.dispersion_crossfade_ms > 500.0f) {
        std::cerr << "Fehler: Dispersion-Crossfade muss zwischen 1 und 500 ms liegen.\n";
        return 1;
    }
    if (cfg.damping_cutoff > static_cast<float>(sample_rate * 0.49)) {
        std::cerr << "Fehler: Daempfungs-Cutoff liegt ueber 0.49 * Sample-Rate der Eingabe.\n";
        return 1;
    }
    if (cfg.dispersion_freq > static_cast<float>(sample_rate * 0.48)) {
        std::cerr << "Fehler: Dispersionsfrequenz liegt ueber 0.48 * Sample-Rate der Eingabe.\n";
        return 1;
    }

    // Engine initialisieren
    dtqw::DTQWReverb<> engine;
    engine.prepare(sample_rate);

    // Reverb Parameter setzen (sofort wirksam für Offline-Rendering)
    engine.setRoomSize(cfg.room_size, true);
    engine.setDecayTime(cfg.decay_t60);
    engine.setDampingCutoff(cfg.damping_cutoff);
    engine.setDryWet(cfg.dry_wet);
    engine.setCoinPhase(cfg.coin_phase_rad, true);
    engine.setChorusRate(cfg.chorus_rate_hz);
    engine.setChorusDepth(cfg.chorus_depth_samples, true);
    engine.setHadamardRouting(cfg.hadamard_in_l, cfg.hadamard_in_r, cfg.hadamard_out_l, cfg.hadamard_out_r);

    if (cfg.dispersion_enabled) {
        engine.setDispersionCrossfadeTime(cfg.dispersion_crossfade_ms);
        if (cfg.dispersion_freq > 0.0f) {
            engine.setDispersionFrequency(cfg.dispersion_freq, true);
        } else {
            engine.setDispersion(cfg.dispersion_g, true);
        }
    } else {
        engine.setDispersionEnabled(false, true);
    }

    if (!cfg.quiet) {
        std::cout << "DSP-KONFIGURATION:\n"
                  << "  Raumgröße:     " << cfg.room_size << "x\n"
                  << "  Decay (T60):   " << cfg.decay_t60 << " s\n"
                  << "  Dämpfung:      " << cfg.damping_cutoff << " Hz\n"
                  << "  Dry/Wet:       " << static_cast<int>(cfg.dry_wet * 100.0f) << " %\n"
                  << "  Coin Phase:    " << cfg.coin_phase_rad << " rad ("
                                      << std::fixed << std::setprecision(1) << (cfg.coin_phase_rad * 180.0f / M_PI) << "°)\n";
        if (cfg.dispersion_enabled) {
            if (cfg.dispersion_freq > 0.0f) {
                std::cout << "  Dispersion:    Aktiviert (Freq: " << cfg.dispersion_freq << " Hz, g="
                          << std::setprecision(3) << engine.getDispersion() << ")\n";
            } else {
                std::cout << "  Dispersion:    Aktiviert (g = " << cfg.dispersion_g << ")\n";
            }
        } else {
            std::cout << "  Dispersion:    Deaktiviert (linear-phasig)\n";
        }
        std::cout << "  Chorus LFO:    " << cfg.chorus_rate_hz << " Hz, Tiefe: " << cfg.chorus_depth_samples << " Samples\n"
                  << "  Hadamard I/O:  In [" << cfg.hadamard_in_l << ", " << cfg.hadamard_in_r
                  << "] -> Out [" << cfg.hadamard_out_l << ", " << cfg.hadamard_out_r << "]\n"
                  << "  Tail-Länge:    " << std::setprecision(2) << cfg.tail_seconds << " s\n"
                  << "  Ausgabe-Bit:   " << (cfg.bit_depth == dtqw::WavBitDepth::Float32 ? "32-bit Float" :
                                         (cfg.bit_depth == dtqw::WavBitDepth::Bit24 ? "24-bit PCM" : "16-bit PCM")) << "\n"
                  << "-------------------------------------------------------------------------------\n";
    }

    size_t tail_frames = cfg.is_ir_mode ? 0 : static_cast<size_t>(cfg.tail_seconds * sample_rate);
    size_t total_render_frames = cfg.is_ir_mode ? static_cast<size_t>(cfg.ir_duration * sample_rate) : (input_frames + tail_frames);

    std::vector<float> in_buf_l(cfg.block_size, 0.0f);
    std::vector<float> in_buf_r(cfg.block_size, 0.0f);
    std::vector<float> out_buf_l(cfg.block_size, 0.0f);
    std::vector<float> out_buf_r(cfg.block_size, 0.0f);

    dtqw::WavStats in_stats;
    dtqw::WavStats out_stats;

    float gain_linear = std::pow(10.0f, cfg.gain_db / 20.0f);

    // Wenn normalisiert werden soll, puffern wir im RAM (Audio-Buffer)
    std::vector<float> full_out_l;
    std::vector<float> full_out_r;
    if (cfg.normalize) {
        full_out_l.reserve(total_render_frames);
        full_out_r.reserve(total_render_frames);
    }

    dtqw::WavWriter writer;
    if (!cfg.normalize) {
        if (!writer.open(cfg.output_path, static_cast<uint32_t>(sample_rate), 2, cfg.bit_depth)) {
            std::cerr << "Fehler beim Erstellen der Ausgabedatei: " << writer.getError() << "\n";
            return 1;
        }
    }

    if (!cfg.quiet) {
        std::cout << "Rendering " << total_render_frames << " Samples ("
                  << std::fixed << std::setprecision(2) << (total_render_frames / sample_rate) << " s) ... " << std::flush;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    size_t frames_processed = 0;
    bool ir_impulse_fired = false;

    while (frames_processed < total_render_frames) {
        size_t chunk_size = std::min(cfg.block_size, total_render_frames - frames_processed);

        if (cfg.is_ir_mode) {
            std::fill(in_buf_l.begin(), in_buf_l.begin() + chunk_size, 0.0f);
            std::fill(in_buf_r.begin(), in_buf_r.begin() + chunk_size, 0.0f);
            if (!ir_impulse_fired) {
                // Dirac delta impulse: 1.0 on both input channels
                in_buf_l[0] = 1.0f;
                in_buf_r[0] = 1.0f;
                ir_impulse_fired = true;
            }
        } else {
            size_t read_count = 0;
            if (reader.getFramesRemaining() > 0) {
                read_count = reader.readFrames(in_buf_l.data(), in_buf_r.data(), chunk_size);
                in_stats.update(in_buf_l.data(), in_buf_r.data(), read_count);
            }
            // Pad rest of chunk with silence (for reverb tail)
            if (read_count < chunk_size) {
                std::fill(in_buf_l.begin() + read_count, in_buf_l.begin() + chunk_size, 0.0f);
                std::fill(in_buf_r.begin() + read_count, in_buf_r.begin() + chunk_size, 0.0f);
            }
        }

        // DSP Processing
        for (size_t i = 0; i < chunk_size; ++i) {
            auto [yl, yr] = engine.processSample(in_buf_l[i], in_buf_r[i]);
            out_buf_l[i] = yl * gain_linear;
            out_buf_r[i] = yr * gain_linear;
        }

        out_stats.update(out_buf_l.data(), out_buf_r.data(), chunk_size);

        if (cfg.normalize) {
            full_out_l.insert(full_out_l.end(), out_buf_l.begin(), out_buf_l.begin() + chunk_size);
            full_out_r.insert(full_out_r.end(), out_buf_r.begin(), out_buf_r.begin() + chunk_size);
        } else {
            writer.writeFrames(out_buf_l.data(), out_buf_r.data(), chunk_size);
        }

        frames_processed += chunk_size;
    }

    in_stats.finalize();
    out_stats.finalize();

    // Normalisierung anwenden falls gewünscht
    if (cfg.normalize) {
        float current_peak = std::max(out_stats.peak_linear_l, out_stats.peak_linear_r);
        float target_linear = std::pow(10.0f, cfg.normalize_target_dbfs / 20.0f);
        float norm_factor = (current_peak > 1e-6f) ? (target_linear / current_peak) : 1.0f;

        if (!writer.open(cfg.output_path, static_cast<uint32_t>(sample_rate), 2, cfg.bit_depth)) {
            std::cerr << "Fehler beim Erstellen der Ausgabedatei: " << writer.getError() << "\n";
            return 1;
        }

        for (size_t i = 0; i < full_out_l.size(); ++i) {
            full_out_l[i] *= norm_factor;
            full_out_r[i] *= norm_factor;
        }

        // Re-compute stats after normalization
        out_stats = dtqw::WavStats{};
        out_stats.update(full_out_l.data(), full_out_r.data(), full_out_l.size());
        out_stats.finalize();

        // Write in blocks
        size_t written = 0;
        while (written < full_out_l.size()) {
            size_t n = std::min(cfg.block_size, full_out_l.size() - written);
            writer.writeFrames(full_out_l.data() + written, full_out_r.data() + written, n);
            written += n;
        }
    }

    writer.close();
    if (!cfg.is_ir_mode) {
        reader.close();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double audio_duration_s = total_render_frames / sample_rate;
    double speedup = (audio_duration_s * 1000.0) / std::max(elapsed_ms, 0.001);

    if (!cfg.quiet) {
        std::cout << "FERTIG!\n\n"
                  << "MESSWERTE & STATISTIK:\n"
                  << "  Rechenzeit:    " << std::fixed << std::setprecision(1) << elapsed_ms << " ms ("
                  << std::setprecision(1) << speedup << "x Echtzeit)\n"
                  << "  Ausgabe-Datei: " << cfg.output_path << "\n";
        if (!cfg.is_ir_mode) {
            std::cout << "  Eingang Peak:  " << std::setprecision(2) << in_stats.getMaxPeakDbfs() << " dBFS (RMS: "
                      << in_stats.getMaxRmsDbfs() << " dBFS)\n";
        }
        std::cout << "  Ausgang Peak:  " << std::setprecision(2) << out_stats.getMaxPeakDbfs() << " dBFS (RMS: "
                  << out_stats.getMaxRmsDbfs() << " dBFS)\n"
                  << "  Crest-Faktor:  " << std::setprecision(2) << out_stats.getCrestFactorDb() << " dB\n";

        if (out_stats.getMaxPeakDbfs() > 0.0f && cfg.bit_depth != dtqw::WavBitDepth::Float32) {
            std::cout << "\n  [!] WARNUNG: Peak überschreitet 0.0 dBFS ("
                      << out_stats.getMaxPeakDbfs() << " dBFS). Bei "
                      << (cfg.bit_depth == dtqw::WavBitDepth::Bit24 ? "24-bit" : "16-bit")
                      << " PCM tritt Clipping auf! Nutze --normalize oder --gain-db.\n";
        }
        std::cout << "===============================================================================\n\n";
    }

    return 0;
}
