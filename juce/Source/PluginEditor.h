#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

namespace ui
{
    const juce::Colour bg      { 0xff0e1014 },
                       panel   { 0xff171a20 },
                       inset   { 0xff0b0d10 },
                       line    { 0xff2a2f38 },
                       text    { 0xffd9dde3 },
                       muted   { 0xff8a919c },
                       cyan    { 0xff38bdf8 },
                       indigo  { 0xff818cf8 },
                       amber   { 0xfff0a83a },
                       emerald { 0xff10b981 };
}

class AxiomLookAndFeel : public juce::LookAndFeel_V4
{
public:
    AxiomLookAndFeel();
    void drawRotarySlider (juce::Graphics&, int x, int y, int w, int h, float pos, float a0, float a1, juce::Slider&) override;
    void drawToggleButton (juce::Graphics&, juce::ToggleButton&, bool highlighted, bool down) override;
    void drawComboBox (juce::Graphics&, int w, int h, bool down, int bx, int by, int bw, int bh, juce::ComboBox&) override;
    juce::Font getComboBoxFont (juce::ComboBox&) override { return juce::Font (juce::FontOptions (11.5f)); }
    juce::Font getPopupMenuFont() override { return juce::Font (juce::FontOptions (13.0f)); }
    juce::Label* createSliderTextBox (juce::Slider&) override;
    void drawButtonBackground (juce::Graphics&, juce::Button&, const juce::Colour&, bool, bool) override;
    void drawScrollbar (juce::Graphics&, juce::ScrollBar&, int x, int y, int w, int h, bool vertical, int thumbStart, int thumbSize, bool over, bool down) override;
    int getDefaultScrollbarWidth() override { return 12; }
};

class Knob : public juce::Component
{
public:
    Knob (juce::AudioProcessorValueTreeState&, const juce::String& id, const juce::String& caption, bool isCyan = true);
    void resized() override;
    juce::Slider slider;
    juce::Label label;
private:
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> att;
};

class Choice : public juce::Component
{
public:
    Choice (juce::AudioProcessorValueTreeState&, const juce::String& id, const juce::String& caption);
    void resized() override;
    juce::ComboBox box;
    juce::Label label;
private:
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> att;
};

class Switch : public juce::ToggleButton
{
public:
    Switch (juce::AudioProcessorValueTreeState&, const juce::String& id, const juce::String& caption);
private:
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> att;
};

class ScopeView : public juce::Component, private juce::Timer
{
public:
    explicit ScopeView (AxiomDTQWProcessor&);
    void paint (juce::Graphics&) override;
private:
    void timerCallback() override;
    AxiomDTQWProcessor& proc;
    std::vector<float> displayBuffer;
};

class ComplexPlaneView : public juce::Component, private juce::Timer
{
public:
    explicit ComplexPlaneView (AxiomDTQWProcessor&);
    void paint (juce::Graphics&) override;
private:
    void timerCallback() override { repaint(); }
    AxiomDTQWProcessor& proc;
};

class Content : public juce::Component
{
public:
    explicit Content (AxiomDTQWProcessor&);
    void paint (juce::Graphics&) override;
    void resized() override;
    int preferredHeight (int width);

private:
    Knob* knob (const char* id, const char* caption, bool isCyan = true);
    Choice* choice (const char* id, const char* caption);
    Switch* toggle (const char* id, const char* caption);

    AxiomDTQWProcessor& proc;
    std::vector<std::unique_ptr<juce::Component>> controls;

    ScopeView scopeView;
    ComplexPlaneView complexPlane;

    juce::TextButton btnGrover { "Grover (0°)" };
    juce::TextButton btnTrapPos { "Trap (+90°)" };
    juce::TextButton btnTrapNeg { "Trap (-90°)" };
};

class AxiomDTQWEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit AxiomDTQWEditor (AxiomDTQWProcessor&);
    ~AxiomDTQWEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    void refreshPresetBox();

    AxiomDTQWProcessor& proc;
    AxiomLookAndFeel lnf;

    juce::ComboBox presets;
    juce::TextButton prev { "<" }, next { ">" };
    juce::Viewport viewport;
    Content content;

    int shownProgram = -1;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AxiomDTQWEditor)
};
