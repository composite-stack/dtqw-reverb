#pragma once
#include <JuceHeader.h>
#include "Parameters.h"
#include "FactoryPresets.h"
#include "DTQWReverb.hpp"
#include <atomic>
#include <array>
#include <unordered_map>

struct ScopeFifo
{
    static constexpr int size = 8192;
    void push (float v) noexcept
    {
        const int w = write.load (std::memory_order_relaxed);
        data[(size_t) w] = v;
        write.store ((w + 1) & (size - 1), std::memory_order_release);
    }
    void copyLatest (float* dst, int n) const noexcept
    {
        int r = write.load (std::memory_order_acquire) - n;
        for (int i = 0; i < n; ++i)
            dst[i] = data[(size_t) ((r + i) & (size - 1))];
    }
    std::array<float, size> data {};
    std::atomic<int> write { 0 };
};

class AxiomDTQWProcessor : public juce::AudioProcessor, private juce::AsyncUpdater
{
public:
    AxiomDTQWProcessor();
    ~AxiomDTQWProcessor() override = default;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout&) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using juce::AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 30.0; }

    int getNumPrograms() override;
    int getCurrentProgram() override { return currentProgram; }
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int) override;

    juce::String presetNote (int index) const;

    juce::AudioProcessorValueTreeState apvts;
    dtqw::DTQWReverb<dtqw::simd::DefaultBackend> core;
    ScopeFifo scopeOut;

private:
    void handleAsyncUpdate() override;
    float getParam (const char* id) const;

    std::unordered_map<std::string, std::atomic<float>*> raw;
    int currentProgram = 0;
    double currentSampleRate = 48000.0;

    int lastInL = -1, lastInR = -1, lastOutL = -1, lastOutR = -1;
    bool lastDispOn = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AxiomDTQWProcessor)
};
