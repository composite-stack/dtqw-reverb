#include "PluginEditor.h"
#include "ReleaseVersion.h"
#include <cmath>

using namespace ui;

static juce::Font font (float h, bool bold = false)
{
    return juce::Font (juce::FontOptions (h, bold ? juce::Font::bold : juce::Font::plain));
}

// ---- LookAndFeel ---------------------------------------------------------------------------
AxiomLookAndFeel::AxiomLookAndFeel()
{
    setColour (juce::ResizableWindow::backgroundColourId, bg);
    setColour (juce::Slider::textBoxTextColourId, text);
    setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    setColour (juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
    setColour (juce::Slider::textBoxHighlightColourId, cyan.withAlpha (0.35f));
    setColour (juce::Label::textColourId, muted);
    setColour (juce::TextEditor::textColourId, text);
    setColour (juce::TextEditor::backgroundColourId, inset);
    setColour (juce::TextEditor::highlightColourId, cyan.withAlpha (0.35f));
    setColour (juce::ComboBox::backgroundColourId, inset);
    setColour (juce::ComboBox::textColourId, text);
    setColour (juce::ComboBox::outlineColourId, line);
    setColour (juce::ComboBox::arrowColourId, muted);
    setColour (juce::PopupMenu::backgroundColourId, panel);
    setColour (juce::PopupMenu::textColourId, text);
    setColour (juce::PopupMenu::highlightedBackgroundColourId, cyan.withAlpha (0.2f));
    setColour (juce::PopupMenu::highlightedTextColourId, text);
}

void AxiomLookAndFeel::drawRotarySlider (juce::Graphics& g, int x, int y, int width, int height,
                                         float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                                         juce::Slider& slider)
{
    auto bounds = juce::Rectangle<int> (x, y, width, height).toFloat();
    auto radius = juce::jmin (bounds.getWidth(), bounds.getHeight()) * 0.5f - 4.0f;
    auto center = bounds.getCentre();

    if (radius < 4.0f) return;

    // Background track circle
    g.setColour (inset);
    g.fillEllipse (center.x - radius, center.y - radius, radius * 2.0f, radius * 2.0f);
    g.setColour (line);
    g.drawEllipse (center.x - radius, center.y - radius, radius * 2.0f, radius * 2.0f, 1.5f);

    // Track arc
    juce::Path trackPath;
    trackPath.addCentredArc (center.x, center.y, radius - 3.0f, radius - 3.0f, 0.0f,
                             rotaryStartAngle, rotaryEndAngle, true);
    g.setColour (panel.brighter (0.1f));
    g.strokePath (trackPath, juce::PathStrokeType (3.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    // Active arc
    float angle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);
    juce::Path activePath;
    activePath.addCentredArc (center.x, center.y, radius - 3.0f, radius - 3.0f, 0.0f,
                              rotaryStartAngle, angle, true);

    auto activeCol = slider.findColour (juce::Slider::rotarySliderFillColourId);
    if (activeCol.isTransparent()) activeCol = cyan;

    g.setColour (activeCol);
    g.strokePath (activePath, juce::PathStrokeType (3.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    // Needle pointer
    juce::Path pointer;
    auto pLen = radius - 5.0f;
    pointer.startNewSubPath (center.x + std::sin (angle) * (pLen * 0.35f),
                             center.y - std::cos (angle) * (pLen * 0.35f));
    pointer.lineTo (center.x + std::sin (angle) * pLen,
                    center.y - std::cos (angle) * pLen);
    g.setColour (text);
    g.strokePath (pointer, juce::PathStrokeType (2.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    // Center pivot dot
    g.setColour (text.withAlpha (0.7f));
    g.fillEllipse (center.x - 2.5f, center.y - 2.5f, 5.0f, 5.0f);
}

void AxiomLookAndFeel::drawToggleButton (juce::Graphics& g, juce::ToggleButton& b, bool /*highlighted*/, bool /*down*/)
{
    auto bounds = b.getLocalBounds().toFloat();
    float h = juce::jmin (bounds.getHeight(), 20.0f);
    float w = 36.0f;
    auto pillBounds = juce::Rectangle<float> (0.0f, (bounds.getHeight() - h) * 0.5f, w, h);

    bool state = b.getToggleState();
    g.setColour (state ? cyan.withAlpha (0.25f) : inset);
    g.fillRoundedRectangle (pillBounds, h * 0.5f);
    g.setColour (state ? cyan : line);
    g.drawRoundedRectangle (pillBounds, h * 0.5f, 1.2f);

    float thumbR = (h - 4.0f) * 0.5f;
    float thumbX = state ? (pillBounds.getRight() - thumbR * 2.0f - 2.0f) : (pillBounds.getX() + 2.0f);
    float thumbY = pillBounds.getY() + 2.0f;

    g.setColour (state ? cyan : muted);
    g.fillEllipse (thumbX, thumbY, thumbR * 2.0f, thumbR * 2.0f);

    g.setFont (font (11.5f));
    g.setColour (state ? text : muted);
    g.drawFittedText (b.getButtonText(), (int) (w + 8.0f), 0, (int) (bounds.getWidth() - w - 8.0f), (int) bounds.getHeight(),
                      juce::Justification::centredLeft, 1);
}

void AxiomLookAndFeel::drawComboBox (juce::Graphics& g, int w, int h, bool /*down*/,
                                     int /*bx*/, int /*by*/, int /*bw*/, int /*bh*/, juce::ComboBox& box)
{
    auto b = juce::Rectangle<float> (0.0f, 0.0f, (float) w, (float) h);
    g.setColour (inset);
    g.fillRoundedRectangle (b, 4.0f);
    g.setColour (box.hasKeyboardFocus (true) ? cyan : line);
    g.drawRoundedRectangle (b, 4.0f, 1.0f);

    // Arrow
    juce::Path arrow;
    float ax = (float) w - 14.0f, ay = (float) h * 0.5f - 2.0f;
    arrow.startNewSubPath (ax - 4.0f, ay);
    arrow.lineTo (ax, ay + 4.0f);
    arrow.lineTo (ax + 4.0f, ay);
    g.setColour (muted);
    g.strokePath (arrow, juce::PathStrokeType (1.5f));
}

juce::Label* AxiomLookAndFeel::createSliderTextBox (juce::Slider& slider)
{
    auto* l = new juce::Label();
    l->setFont (font (11.5f, true));
    l->setJustificationType (juce::Justification::centred);
    l->setColour (juce::Label::textColourId, text);
    l->setColour (juce::Label::textWhenEditingColourId, text);
    l->setColour (juce::Label::backgroundWhenEditingColourId, inset);
    l->setColour (juce::Label::outlineWhenEditingColourId, cyan);
    return l;
}

void AxiomLookAndFeel::drawButtonBackground (juce::Graphics& g, juce::Button& b, const juce::Colour& /*bgCol*/,
                                            bool isMouseOver, bool isButtonDown)
{
    auto r = b.getLocalBounds().toFloat();
    g.setColour (isButtonDown ? cyan.withAlpha (0.3f) : isMouseOver ? panel.brighter (0.2f) : inset);
    g.fillRoundedRectangle (r, 4.0f);
    g.setColour (isMouseOver ? cyan : line);
    g.drawRoundedRectangle (r, 4.0f, 1.0f);
}

void AxiomLookAndFeel::drawScrollbar (juce::Graphics& g, juce::ScrollBar&, int x, int y, int w, int h,
                                      bool /*vertical*/, int thumbStart, int thumbSize, bool isMouseOver, bool /*down*/)
{
    g.setColour (juce::Colour (0xff0f172a));
    g.fillRect (x, y, w, h);

    if (thumbSize > 0)
    {
        auto thumbBounds = juce::Rectangle<float> ((float) x + 2.0f, (float) thumbStart,
                                                  (float) w - 4.0f, (float) thumbSize);
        g.setColour (isMouseOver ? cyan.withAlpha (0.6f) : juce::Colour (0xff334155));
        g.fillRoundedRectangle (thumbBounds, 3.0f);
    }
}

// ---- Custom Components ---------------------------------------------------------------------
Knob::Knob (juce::AudioProcessorValueTreeState& apvts, const juce::String& id, const juce::String& caption, bool isCyan)
{
    slider.setSliderStyle (juce::Slider::RotaryVerticalDrag);
    slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 76, 16);
    slider.setMouseWheelRunsSlider (false); // Don't steal mouse wheel from viewport!
    slider.setColour (juce::Slider::rotarySliderFillColourId, isCyan ? cyan : amber);
    addAndMakeVisible (slider);

    label.setText (caption, juce::dontSendNotification);
    label.setFont (font (11.0f));
    label.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (label);

    att = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (apvts, id, slider);
}

void Knob::resized()
{
    auto b = getLocalBounds();
    label.setBounds (b.removeFromBottom (16));
    slider.setBounds (b);
}

Choice::Choice (juce::AudioProcessorValueTreeState& apvts, const juce::String& id, const juce::String& caption)
{
    box.setMouseWheelRunsComboBox (false);
    if (const auto* spec = findChoice (id))
    {
        for (size_t i = 0; i < spec->labels.size(); ++i)
            box.addItem (U (spec->labels[i]), (int) i + 1);
    }
    addAndMakeVisible (box);

    label.setText (caption, juce::dontSendNotification);
    label.setFont (font (11.0f));
    label.setJustificationType (juce::Justification::centredLeft);
    addAndMakeVisible (label);

    att = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (apvts, id, box);
}

void Choice::resized()
{
    auto b = getLocalBounds();
    label.setBounds (b.removeFromLeft (90));
    box.setBounds (b);
}

Switch::Switch (juce::AudioProcessorValueTreeState& apvts, const juce::String& id, const juce::String& caption)
{
    setButtonText (caption);
    att = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (apvts, id, *this);
}

ScopeView::ScopeView (AxiomDTQWProcessor& p) : proc (p)
{
    displayBuffer.resize (256, 0.0f);
    startTimerHz (30);
}

void ScopeView::timerCallback()
{
    proc.scopeOut.copyLatest (displayBuffer.data(), (int) displayBuffer.size());
    repaint();
}

void ScopeView::paint (juce::Graphics& g)
{
    auto b = getLocalBounds().toFloat();
    g.setColour (inset);
    g.fillRoundedRectangle (b, 6.0f);
    g.setColour (line);
    g.drawRoundedRectangle (b, 6.0f, 1.0f);

    // Center reference line
    float midY = b.getCentreY();
    g.setColour (panel.brighter (0.1f));
    g.drawHorizontalLine ((int) midY, b.getX() + 4.0f, b.getRight() - 4.0f);

    juce::Path wavePath;
    float stepX = (b.getWidth() - 8.0f) / (float) (displayBuffer.size() - 1);
    float startX = b.getX() + 4.0f;

    for (size_t i = 0; i < displayBuffer.size(); ++i)
    {
        float x = startX + (float) i * stepX;
        float y = midY - displayBuffer[i] * (b.getHeight() * 0.45f);
        y = juce::jlimit (b.getY() + 2.0f, b.getBottom() - 2.0f, y);

        if (i == 0) wavePath.startNewSubPath (x, y);
        else        wavePath.lineTo (x, y);
    }

    g.setColour (cyan.withAlpha (0.85f));
    g.strokePath (wavePath, juce::PathStrokeType (1.5f, juce::PathStrokeType::curved));

    g.setFont (font (10.0f));
    g.setColour (muted);
    g.drawText ("AUSGANGS-OSZILLOSKOP", b.reduced (8, 4), juce::Justification::topRight);
}

ComplexPlaneView::ComplexPlaneView (AxiomDTQWProcessor& p) : proc (p)
{
    startTimerHz (15);
}

void ComplexPlaneView::paint (juce::Graphics& g)
{
    auto b = getLocalBounds().toFloat();
    g.setColour (inset);
    g.fillRoundedRectangle (b, 6.0f);
    g.setColour (line);
    g.drawRoundedRectangle (b, 6.0f, 1.0f);

    auto center = b.getCentre();
    float r = juce::jmin (b.getWidth(), b.getHeight()) * 0.38f;

    // Complex Unit Circle
    g.setColour (panel.brighter (0.15f));
    g.drawEllipse (center.x - r, center.y - r, r * 2.0f, r * 2.0f, 1.2f);
    g.drawHorizontalLine ((int) center.y, center.x - r - 4.0f, center.x + r + 4.0f);
    g.drawVerticalLine ((int) center.x, center.y - r - 4.0f, center.y + r + 4.0f);

    // Static eigenvalues at +1 and -1 (Walsh subspace e0 and e1)
    g.setColour (emerald);
    g.fillEllipse (center.x + r - 3.5f, center.y - 3.5f, 7.0f, 7.0f); // +1 (e0 Injektion)
    g.fillEllipse (center.x - r - 3.5f, center.y - 3.5f, 7.0f, 7.0f); // -1 (e1 Auslesung)

    // Moving eigenvalues at -exp(+-i theta)
    float theta = 0.0f;
    if (auto* p = proc.apvts.getRawParameterValue ("coinPhase"))
        theta = p->load();

    float ex1_x = center.x - r * std::cos (theta);
    float ex1_y = center.y - r * std::sin (theta);
    float ex2_x = center.x - r * std::cos (-theta);
    float ex2_y = center.y - r * std::sin (-theta);

    g.setColour (cyan);
    g.fillEllipse (ex1_x - 4.0f, ex1_y - 4.0f, 8.0f, 8.0f);
    g.fillEllipse (ex2_x - 4.0f, ex2_y - 4.0f, 8.0f, 8.0f);

    g.setFont (font (9.5f));
    g.setColour (muted);
    g.drawText ("EIGENWERTE", (int) b.getX() + 6, (int) b.getY() + 4, 80, 14, juce::Justification::topLeft);

    juce::String phaseStr = juce::String (theta, 2) + " rad";
    g.drawText (phaseStr, (int) b.getRight() - 76, (int) b.getY() + 4, 70, 14, juce::Justification::topRight);
}

// ---- Content -------------------------------------------------------------------------------
Content::Content (AxiomDTQWProcessor& p) : proc (p), scopeView (p), complexPlane (p)
{
    addAndMakeVisible (scopeView);
    addAndMakeVisible (complexPlane);

    btnGrover.onClick = [this] {
        if (auto* param = proc.apvts.getParameter ("coinPhase"))
            param->setValueNotifyingHost (param->convertTo0to1 (0.0f));
    };
    btnTrapPos.onClick = [this] {
        if (auto* param = proc.apvts.getParameter ("coinPhase"))
            param->setValueNotifyingHost (param->convertTo0to1 (1.570796f));
    };
    btnTrapNeg.onClick = [this] {
        if (auto* param = proc.apvts.getParameter ("coinPhase"))
            param->setValueNotifyingHost (param->convertTo0to1 (-1.570796f));
    };
    addAndMakeVisible (btnGrover);
    addAndMakeVisible (btnTrapPos);
    addAndMakeVisible (btnTrapNeg);

    // Knobs
    knob ("coinPhase",    "Phase θ", true);
    knob ("roomSize",     "Größe",   true);
    knob ("decay",        "T60",     true);
    knob ("cutoff",       "Dämpfung", true);
    knob ("chorusDepth",  "Tiefe",   false);
    knob ("chorusRate",   "Rate",    false);
    knob ("dispersionG",  "Allpass g", false);
    knob ("mix",          "Mix",     true);

    // Switches
    toggle ("dispersionOn", "Dispersion aktiv");

    // Choices
    choice ("inLRow",  "In L (Walsh)");
    choice ("inRRow",  "In R (Walsh)");
    choice ("outLRow", "Out L (Walsh)");
    choice ("outRRow", "Out R (Walsh)");
}

Knob* Content::knob (const char* id, const char* caption, bool isCyan)
{
    auto k = std::make_unique<Knob> (proc.apvts, id, U (caption), isCyan);
    auto* ptr = k.get();
    controls.push_back (std::move (k));
    addAndMakeVisible (ptr);
    return ptr;
}

Choice* Content::choice (const char* id, const char* caption)
{
    auto c = std::make_unique<Choice> (proc.apvts, id, U (caption));
    auto* ptr = c.get();
    controls.push_back (std::move (c));
    addAndMakeVisible (ptr);
    return ptr;
}

Switch* Content::toggle (const char* id, const char* caption)
{
    auto s = std::make_unique<Switch> (proc.apvts, id, U (caption));
    auto* ptr = s.get();
    controls.push_back (std::move (s));
    addAndMakeVisible (ptr);
    return ptr;
}

int Content::preferredHeight (int /*width*/)
{
    return 580;
}

void Content::paint (juce::Graphics& g)
{
    g.fillAll (bg);

    // Helper to draw section panel
    auto drawSection = [&g] (juce::Rectangle<int> r, const char* title, const juce::Colour& accent) {
        g.setColour (panel);
        g.fillRoundedRectangle (r.toFloat(), 6.0f);
        g.setColour (line);
        g.drawRoundedRectangle (r.toFloat(), 6.0f, 1.0f);

        auto th = r.removeFromTop (26);
        g.setColour (accent);
        g.fillRect (th.getX() + 8, th.getY() + 7, 3, 12);
        g.setFont (font (11.0f, true));
        g.setColour (text);
        g.drawText (U (title), th.getX() + 16, th.getY(), th.getWidth() - 20, th.getHeight(), juce::Justification::centredLeft);
        g.setColour (line);
        g.drawHorizontalLine (th.getBottom(), (float) th.getX(), (float) th.getRight());
    };

    int w = getWidth();

    // Section bounds matching resized()
    int pad = 12;
    int colW = (w - pad * 4) / 3;

    drawSection ({ pad, pad, colW, 260 }, "1. QUANTUM COIN SCATTERING", cyan);
    drawSection ({ pad * 2 + colW, pad, colW, 260 }, "2. GEOMETRIE & 4D HYPERCUBE", cyan);
    drawSection ({ pad * 3 + colW * 2, pad, colW, 260 }, "3. NACHHALL & TPT-DÄMPFUNG", cyan);

    int y2 = pad + 270;
    drawSection ({ pad, y2, colW, 280 }, "4. 32-LFO CHORUS MODULATION", amber);
    drawSection ({ pad * 2 + colW, y2, colW, 280 }, "5. FREQUENZ-DISPERSION", amber);
    drawSection ({ pad * 3 + colW * 2, y2, colW, 280 }, "6. AUSKOPPLUNG & WALSH-ROUTING", cyan);
}

void Content::resized()
{
    int w = getWidth();
    int pad = 12;
    int colW = (w - pad * 4) / 3;

    // Section 1: Quantum Coin
    auto sec1 = juce::Rectangle<int> (pad, pad, colW, 260).reduced (10, 32);
    auto sec1Left = sec1.removeFromLeft (sec1.getWidth() / 2);
    controls[0]->setBounds (sec1Left.removeFromTop (120)); // Coin Phase Knob
    auto btnArea = sec1Left;
    btnGrover.setBounds (btnArea.removeFromTop (24).reduced (2, 2));
    btnTrapPos.setBounds (btnArea.removeFromTop (24).reduced (2, 2));
    btnTrapNeg.setBounds (btnArea.removeFromTop (24).reduced (2, 2));
    complexPlane.setBounds (sec1.reduced (4));

    // Section 2: Geometry & Hypercube
    auto sec2 = juce::Rectangle<int> (pad * 2 + colW, pad, colW, 260).reduced (10, 32);
    controls[1]->setBounds (sec2.removeFromLeft (110).removeFromTop (130)); // Room Size Knob

    // Section 3: Decay & Damping
    auto sec3 = juce::Rectangle<int> (pad * 3 + colW * 2, pad, colW, 260).reduced (10, 32);
    controls[2]->setBounds (sec3.removeFromLeft (sec3.getWidth() / 2).removeFromTop (130)); // Decay Knob
    controls[3]->setBounds (sec3.removeFromTop (130)); // Cutoff Knob

    // Row 2
    int y2 = pad + 270;

    // Section 4: Chorus
    auto sec4 = juce::Rectangle<int> (pad, y2, colW, 280).reduced (10, 32);
    controls[4]->setBounds (sec4.removeFromLeft (sec4.getWidth() / 2).removeFromTop (130)); // Chorus Depth
    controls[5]->setBounds (sec4.removeFromTop (130)); // Chorus Rate

    // Section 5: Dispersion
    auto sec5 = juce::Rectangle<int> (pad * 2 + colW, y2, colW, 280).reduced (10, 32);
    controls[8]->setBounds (sec5.removeFromTop (28)); // Dispersion Toggle
    controls[6]->setBounds (sec5.removeFromLeft (110).removeFromTop (130)); // Dispersion g

    // Section 6: Output & Matrixing
    auto sec6 = juce::Rectangle<int> (pad * 3 + colW * 2, y2, colW, 280).reduced (10, 32);
    auto sec6Top = sec6.removeFromTop (130);
    controls[7]->setBounds (sec6Top.removeFromLeft (100)); // Mix Knob
    scopeView.setBounds (sec6Top.reduced (4));

    auto routeArea = sec6;
    int rH = 24;
    controls[9]->setBounds (routeArea.removeFromTop (rH));
    controls[10]->setBounds (routeArea.removeFromTop (rH));
    controls[11]->setBounds (routeArea.removeFromTop (rH));
    controls[12]->setBounds (routeArea.removeFromTop (rH));
}

// ---- Editor --------------------------------------------------------------------------------
static constexpr int kHeaderH = 46, kDefaultW = 1180, kDefaultH = 640;

AxiomDTQWEditor::AxiomDTQWEditor (AxiomDTQWProcessor& p)
    : AudioProcessorEditor (&p), proc (p), content (p)
{
    setLookAndFeel (&lnf);

    for (int i = 0; i < proc.getNumPrograms(); ++i)
        presets.addItem (proc.getProgramName (i), i + 1);

    presets.setTitle ("Preset");
    presets.onChange = [this] {
        const int i = presets.getSelectedItemIndex();
        if (i >= 0 && i != proc.getCurrentProgram())
            proc.setCurrentProgram (i);
        refreshPresetBox();
    };

    prev.onClick = [this] {
        const int n = proc.getNumPrograms();
        proc.setCurrentProgram ((proc.getCurrentProgram() + n - 1) % n);
        refreshPresetBox();
    };

    next.onClick = [this] {
        proc.setCurrentProgram ((proc.getCurrentProgram() + 1) % proc.getNumPrograms());
        refreshPresetBox();
    };

    addAndMakeVisible (presets);
    addAndMakeVisible (prev);
    addAndMakeVisible (next);

    viewport.setViewedComponent (&content, false);
    viewport.setScrollBarsShown (true, true);
    viewport.getVerticalScrollBar().setAutoHide (false); // Always visible, working scrollbar!
    viewport.setScrollBarThickness (12);
    viewport.setSingleStepSizes (24, 24);
    addAndMakeVisible (viewport);

    setResizable (true, true);
    setResizeLimits (720, 420, 2400, 1600);

    // Fit laptop screens: never open taller than available desktop height
    int h = kDefaultH, w = kDefaultW;
    if (auto* d = juce::Desktop::getInstance().getDisplays().getPrimaryDisplay())
    {
        h = juce::jmin (h, d->userArea.getHeight() - 80);
        w = juce::jmin (w, d->userArea.getWidth() - 40);
    }
    setSize (juce::jmax (720, w), juce::jmax (420, h));

    refreshPresetBox();
    startTimerHz (10);
}

AxiomDTQWEditor::~AxiomDTQWEditor()
{
    setLookAndFeel (nullptr);
}

void AxiomDTQWEditor::refreshPresetBox()
{
    shownProgram = proc.getCurrentProgram();
    presets.setSelectedItemIndex (shownProgram, juce::dontSendNotification);
    repaint (0, 0, getWidth(), kHeaderH);
}

void AxiomDTQWEditor::timerCallback()
{
    if (proc.getCurrentProgram() != shownProgram)
        refreshPresetBox();
}

void AxiomDTQWEditor::paint (juce::Graphics& g)
{
    g.fillAll (bg);

    // Header panel
    auto h = getLocalBounds().removeFromTop (kHeaderH);
    g.setColour (panel);
    g.fillRect (h);
    g.setColour (line);
    g.drawHorizontalLine (kHeaderH - 1, 0.0f, (float) getWidth());

    // Brand logo & title
    g.setFont (font (20.0f, true));
    g.setColour (text);
    g.drawText ("AXIOM", 14, 6, 75, 24, juce::Justification::centredLeft);
    g.setColour (cyan);
    g.drawText ("DTQW", 88, 6, 70, 24, juce::Justification::centredLeft);

    g.setColour (muted);
    g.setFont (font (10.5f));
    g.drawText (U ("QUANTUM WALK REVERB  ·  v" AXIOM_RELEASE_VERSION "  ·  " AXIOM_RELEASE_DATE),
                14, 28, 360, 14, juce::Justification::centredLeft);

    // Preset note text
    const int noteX = 380;
    const int noteW = presets.getX() - 20 - noteX;
    if (noteW > 100)
    {
        g.setFont (font (11.0f));
        g.setColour (text.withAlpha (0.9f));
        g.drawFittedText (proc.presetNote (proc.getCurrentProgram()),
                          noteX, 10, noteW, 26, juce::Justification::centredRight, 2, 0.9f);
    }
}

void AxiomDTQWEditor::resized()
{
    auto header = getLocalBounds().removeFromTop (kHeaderH).reduced (10, 10);
    next.setBounds (header.removeFromRight (26));
    header.removeFromRight (4);
    presets.setBounds (header.removeFromRight (juce::jmin (240, juce::jmax (150, getWidth() / 4))));
    header.removeFromRight (4);
    prev.setBounds (header.removeFromRight (26));

    viewport.setBounds (getLocalBounds().withTrimmedTop (kHeaderH));
    const int w = juce::jmax (1100, viewport.getWidth() - viewport.getScrollBarThickness());
    content.setSize (w, content.preferredHeight (w));
}
