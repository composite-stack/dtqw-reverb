#pragma once
#include <JuceHeader.h>
#include <vector>
#include <unordered_map>

struct Preset
{
    const char* name;
    const char* note;
    std::unordered_map<std::string, float> values;
};

inline const std::vector<Preset>& factoryPresets()
{
    static const std::vector<Preset> presets = {
        {
            "Grover Diffuser",
            "Klassischer Grover-Coin: Maximale ballistische Streuung auf dem 4D-Hypercube ohne Eigenwert-Phasendrehung.",
            {
                { "roomSize", 1.0f }, { "decay", 1.8f }, { "cutoff", 5500.0f },
                { "coinPhase", 0.0f }, { "chorusDepth", 3.0f }, { "chorusRate", 0.8f },
                { "dispersionOn", 0.0f }, { "dispersionG", 0.0f }, { "mix", 1.0f },
                { "inLRow", 1.0f }, { "inRRow", 2.0f }, { "outLRow", 5.0f }, { "outRRow", 6.0f }
            }
        },
        {
            "Quantum Trap (+pi/2)",
            "Resonante Interferenz-Knoten: Destruktive Auslöschung transversaler Pfade erzeugt schwebende Moden.",
            {
                { "roomSize", 1.0f }, { "decay", 2.5f }, { "cutoff", 7000.0f },
                { "coinPhase", 1.570796f }, { "chorusDepth", 1.5f }, { "chorusRate", 0.4f },
                { "dispersionOn", 0.0f }, { "dispersionG", 0.0f }, { "mix", 1.0f },
                { "inLRow", 1.0f }, { "inRRow", 2.0f }, { "outLRow", 5.0f }, { "outRRow", 6.0f }
            }
        },
        {
            "Quantum Trap (-pi/2)",
            "Invertierte Phasen-Traps: Dunkle, dichte Resonanzen mit veränderten Walsh-Knoten.",
            {
                { "roomSize", 1.25f }, { "decay", 2.5f }, { "cutoff", 6500.0f },
                { "coinPhase", -1.570796f }, { "chorusDepth", 2.0f }, { "chorusRate", 0.5f },
                { "dispersionOn", 0.0f }, { "dispersionG", 0.0f }, { "mix", 1.0f },
                { "inLRow", 1.0f }, { "inRRow", 2.0f }, { "outLRow", 5.0f }, { "outRRow", 6.0f }
            }
        },
        {
            "Ethereal Chamber",
            "Großer Raum mit sanfter Frequenz-Dispersion und schwebendem 32-LFO-Chorus.",
            {
                { "roomSize", 1.6f }, { "decay", 4.5f }, { "cutoff", 4800.0f },
                { "coinPhase", 0.785398f }, { "chorusDepth", 5.0f }, { "chorusRate", 0.35f },
                { "dispersionOn", 1.0f }, { "dispersionG", 0.45f }, { "mix", 1.0f },
                { "inLRow", 1.0f }, { "inRRow", 2.0f }, { "outLRow", 5.0f }, { "outRRow", 6.0f }
            }
        },
        {
            "Metallic Resonator",
            "Kompakter Hypercube mit negativer Dispersion: Metallische Transienten und dichte Nahfeldreflexionen.",
            {
                { "roomSize", 0.45f }, { "decay", 1.2f }, { "cutoff", 11000.0f },
                { "coinPhase", 2.356194f }, { "chorusDepth", 0.8f }, { "chorusRate", 1.5f },
                { "dispersionOn", 1.0f }, { "dispersionG", -0.60f }, { "mix", 1.0f },
                { "inLRow", 1.0f }, { "inRRow", 2.0f }, { "outLRow", 5.0f }, { "outRRow", 6.0f }
            }
        },
        {
            "Discrete Walk (Pure)",
            "Vollständig unmodulierter Quantum Walk: Reine mathematische Interferenz aller 64 Delay-Pfade.",
            {
                { "roomSize", 1.0f }, { "decay", 2.0f }, { "cutoff", 16000.0f },
                { "coinPhase", 0.0f }, { "chorusDepth", 0.0f }, { "chorusRate", 0.01f },
                { "dispersionOn", 0.0f }, { "dispersionG", 0.0f }, { "mix", 1.0f },
                { "inLRow", 1.0f }, { "inRRow", 2.0f }, { "outLRow", 5.0f }, { "outRRow", 6.0f }
            }
        }
    };
    return presets;
}
