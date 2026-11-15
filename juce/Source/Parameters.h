#pragma once
#include <JuceHeader.h>
#include <vector>
#include <string>

inline juce::String U (const char* utf8) { return juce::String::fromUTF8 (utf8); }

struct ChoiceSpec { const char* id; const char* name; std::vector<const char*> ids; std::vector<const char*> labels; int def; };

inline const std::vector<ChoiceSpec>& choiceSpecs()
{
    static const std::vector<const char*> walshIds = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"
    };
    static const std::vector<const char*> walshLabels = {
        "Knoten 0 (e0)", "Knoten 1 (e1)", "Knoten 2 (e2)", "Knoten 3 (e3)",
        "Knoten 4 (e4)", "Knoten 5 (e5)", "Knoten 6 (e6)", "Knoten 7 (e7)",
        "Knoten 8 (e8)", "Knoten 9 (e9)", "Knoten 10 (e10)", "Knoten 11 (e11)",
        "Knoten 12 (e12)", "Knoten 13 (e13)", "Knoten 14 (e14)", "Knoten 15 (e15)"
    };

    static const std::vector<ChoiceSpec> specs = {
        { "inLRow", "Injektion L", walshIds, walshLabels, 1 },
        { "inRRow", "Injektion R", walshIds, walshLabels, 2 },
        { "outLRow", "Auskopplung L", walshIds, walshLabels, 5 },
        { "outRRow", "Auskopplung R", walshIds, walshLabels, 6 }
    };
    return specs;
}

inline const ChoiceSpec* findChoice (const juce::String& id)
{
    for (auto& c : choiceSpecs())
        if (id == c.id) return &c;
    return nullptr;
}

enum class Fmt { times, seconds, hz, rad, smp, plain, pct };
struct FloatSpec { const char* id; const char* name; float min, max, def; bool log; Fmt fmt; };

inline const std::vector<FloatSpec>& floatSpecs()
{
    static const std::vector<FloatSpec> specs = {
        { "roomSize",     "Raumgröße",       0.1f,     3.5f,     1.0f,   false, Fmt::times },
        { "decay",        "Nachhall T60",    0.1f,     30.0f,    1.8f,   false, Fmt::seconds },
        { "cutoff",       "Dämpfung",        200.0f,   20000.0f, 5500.0f, true,  Fmt::hz },
        { "coinPhase",    "Coin-Phase θ",   -3.14159f, 3.14159f, 0.0f,   false, Fmt::rad },
        { "chorusDepth",  "Chorus-Tiefe",    0.0f,     24.0f,    3.0f,   false, Fmt::smp },
        { "chorusRate",   "Chorus-Rate",     0.01f,    15.0f,    0.8f,   false, Fmt::hz },
        { "dispersionG",  "Allpass g",      -0.90f,    0.90f,    0.0f,   false, Fmt::plain },
        { "mix",          "Wet / Dry",       0.0f,     1.0f,     1.0f,   false, Fmt::pct }
    };
    return specs;
}

inline const FloatSpec* findFloat (const juce::String& id)
{
    for (auto& f : floatSpecs())
        if (id == f.id) return &f;
    return nullptr;
}

inline juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Choice parameters
    for (const auto& c : choiceSpecs())
    {
        juce::StringArray labels;
        for (const auto* l : c.labels) labels.add (U (l));
        params.push_back (std::make_unique<juce::AudioParameterChoice> (
            juce::ParameterID { c.id, 1 }, U (c.name), labels, c.def));
    }

    // Bool switch
    params.push_back (std::make_unique<juce::AudioParameterBool> (
        juce::ParameterID { "dispersionOn", 1 }, U ("Allpass-Dispersion"), false));

    // Float parameters
    for (const auto& f : floatSpecs())
    {
        auto range = f.log ? juce::NormalisableRange<float> (f.min, f.max, 0.01f, 0.25f)
                           : juce::NormalisableRange<float> (f.min, f.max);

        auto stringFromVal = [fmt = f.fmt] (float v, int) -> juce::String {
            switch (fmt)
            {
                case Fmt::times:   return juce::String (v, 2) + " ×";
                case Fmt::seconds: return juce::String (v, 2) + " s";
                case Fmt::hz:      return v >= 1000.0f ? juce::String (v / 1000.0f, 2) + " kHz" : juce::String (juce::roundToInt (v)) + " Hz";
                case Fmt::rad: {
                    float deg = v * 180.0f / 3.14159265f;
                    return juce::String (v, 2) + " rad (" + juce::String (juce::roundToInt (deg)) + "°)";
                }
                case Fmt::smp:     return juce::String (v, 1) + " smp";
                case Fmt::plain:   return juce::String (v, 2);
                case Fmt::pct:     return juce::String (juce::roundToInt (v * 100.0f)) + " %";
            }
            return juce::String (v, 2);
        };

        params.push_back (std::make_unique<juce::AudioParameterFloat> (
            juce::ParameterID { f.id, 1 }, U (f.name), range, f.def,
            juce::AudioParameterFloatAttributes().withStringFromValueFunction (stringFromVal)));
    }

    return { params.begin(), params.end() };
}
