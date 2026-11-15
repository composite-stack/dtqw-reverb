#include "PluginProcessor.h"
#include "PluginEditor.h"

AxiomDTQWProcessor::AxiomDTQWProcessor()
    : AudioProcessor (BusesProperties()
                        .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                        .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "PARAMETERS", createParameterLayout())
{
    for (const auto& c : choiceSpecs())
        raw[c.id] = apvts.getRawParameterValue (c.id);

    raw["dispersionOn"] = apvts.getRawParameterValue ("dispersionOn");

    for (const auto& f : floatSpecs())
        raw[f.id] = apvts.getRawParameterValue (f.id);
}

void AxiomDTQWProcessor::prepareToPlay (double sampleRate, int /*samplesPerBlock*/)
{
    currentSampleRate = sampleRate;
    core.prepare (sampleRate);

    // Initialise core with current parameter values
    core.setRoomSize (getParam ("roomSize"), true);
    core.setDecayTime (getParam ("decay"));
    core.setDampingCutoff (getParam ("cutoff"));
    core.setCoinPhase (getParam ("coinPhase"), true);
    core.setChorusDepth (getParam ("chorusDepth"), true);
    core.setChorusRate (getParam ("chorusRate"));
    core.setDryWet (getParam ("mix"));

    lastDispOn = getParam ("dispersionOn") > 0.5f;
    core.setDispersionEnabled (lastDispOn, true);
    core.setDispersion (getParam ("dispersionG"), true);

    lastInL = (int) getParam ("inLRow");
    lastInR = (int) getParam ("inRRow");
    lastOutL = (int) getParam ("outLRow");
    lastOutR = (int) getParam ("outRRow");
    core.setHadamardRouting (lastInL, lastInR, lastOutL, lastOutR);
}

bool AxiomDTQWProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& mainIn  = layouts.getMainInputChannelSet();
    const auto& mainOut = layouts.getMainOutputChannelSet();

    if (mainOut != juce::AudioChannelSet::stereo() && mainOut != juce::AudioChannelSet::mono())
        return false;

    if (mainIn != juce::AudioChannelSet::stereo() && mainIn != juce::AudioChannelSet::mono() && ! mainIn.isDisabled())
        return false;

    return true;
}

float AxiomDTQWProcessor::getParam (const char* id) const
{
    auto it = raw.find (id);
    return it != raw.end() && it->second != nullptr ? it->second->load (std::memory_order_relaxed) : 0.0f;
}

void AxiomDTQWProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& /*midiMessages*/)
{
    juce::ScopedNoDenormals noDenormals;
    const int numSamples = buffer.getNumSamples();
    if (numSamples == 0) return;

    const int totalInChannels  = getTotalNumInputChannels();
    const int totalOutChannels = getTotalNumOutputChannels();

    // Clear any extra output channels beyond stereo
    for (int ch = totalInChannels; ch < totalOutChannels; ++ch)
        buffer.clear (ch, 0, numSamples);

    // Update continuous parameters (internally smoothed by DTQWReverb)
    core.setRoomSize (getParam ("roomSize"));
    core.setDecayTime (getParam ("decay"));
    core.setDampingCutoff (getParam ("cutoff"));
    core.setCoinPhase (getParam ("coinPhase"));
    core.setChorusDepth (getParam ("chorusDepth"));
    core.setChorusRate (getParam ("chorusRate"));
    core.setDryWet (getParam ("mix"));

    const bool dispOn = getParam ("dispersionOn") > 0.5f;
    if (dispOn != lastDispOn)
    {
        core.setDispersionEnabled (dispOn);
        lastDispOn = dispOn;
    }
    core.setDispersion (getParam ("dispersionG"));

    const int inL  = (int) getParam ("inLRow");
    const int inR  = (int) getParam ("inRRow");
    const int outL = (int) getParam ("outLRow");
    const int outR = (int) getParam ("outRRow");
    if (inL != lastInL || inR != lastInR || outL != lastOutL || outR != lastOutR)
    {
        core.setHadamardRouting (inL, inR, outL, outR);
        lastInL = inL; lastInR = inR; lastOutL = outL; lastOutR = outR;
    }

    // Prepare stereo channel pointers
    float* channelPointers[2];
    if (totalInChannels >= 2)
    {
        channelPointers[0] = buffer.getWritePointer (0);
        channelPointers[1] = buffer.getWritePointer (1);
    }
    else if (totalInChannels == 1)
    {
        channelPointers[0] = buffer.getWritePointer (0);
        channelPointers[1] = totalOutChannels > 1 ? buffer.getWritePointer (1) : buffer.getWritePointer (0);
        if (totalOutChannels > 1)
            buffer.copyFrom (1, 0, buffer.getReadPointer (0), numSamples);
    }
    else
    {
        // Standalone or generator: clear buffer and process
        buffer.clear();
        channelPointers[0] = buffer.getWritePointer (0);
        channelPointers[1] = totalOutChannels > 1 ? buffer.getWritePointer (1) : buffer.getWritePointer (0);
    }

    core.processBlock (channelPointers, 2, (size_t) numSamples);

    // Push samples to UI scope buffer
    for (int i = 0; i < numSamples; ++i)
        scopeOut.push (0.5f * (channelPointers[0][i] + channelPointers[1][i]));
}

int AxiomDTQWProcessor::getNumPrograms() { return (int) factoryPresets().size(); }

const juce::String AxiomDTQWProcessor::getProgramName (int index)
{
    const auto& p = factoryPresets();
    return (index >= 0 && index < (int) p.size()) ? juce::String (p[(size_t) index].name) : juce::String();
}

juce::String AxiomDTQWProcessor::presetNote (int index) const
{
    const auto& p = factoryPresets();
    return (index >= 0 && index < (int) p.size()) ? U (p[(size_t) index].note) : juce::String();
}

void AxiomDTQWProcessor::setCurrentProgram (int index)
{
    const auto& presets = factoryPresets();
    if (index < 0 || index >= (int) presets.size()) return;
    currentProgram = index;
    const auto& pr = presets[(size_t) index];

    for (const auto& kv : pr.values)
    {
        if (auto* p = apvts.getParameter (kv.first))
            p->setValueNotifyingHost (p->convertTo0to1 (kv.second));
    }
    triggerAsyncUpdate();
}

void AxiomDTQWProcessor::handleAsyncUpdate() {}

void AxiomDTQWProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    xml->setAttribute ("currentProgram", currentProgram);
    copyXmlToBinary (*xml, destData);
}

void AxiomDTQWProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml (getXmlFromBinary (data, sizeInBytes));
    if (xml != nullptr && xml->hasTagName (apvts.state.getType()))
    {
        currentProgram = xml->getIntAttribute ("currentProgram", 0);
        apvts.replaceState (juce::ValueTree::fromXml (*xml));
    }
}

juce::AudioProcessorEditor* AxiomDTQWProcessor::createEditor()
{
    return new AxiomDTQWEditor (*this);
}

// Creation function required by JUCE
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AxiomDTQWProcessor();
}
