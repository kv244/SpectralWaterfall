#include "AboutComponent.h"
#include "AppVersion.h"
#include "CudaProcessor.h"

#include <cmath>

// ---------------------------------------------------------------------------
namespace
{
juce::String formatCudaVer (int v)
{
    if (v <= 0)
        return "n/a";
    return juce::String (v / 1000) + "." + juce::String ((v % 1000) / 10);
}
} // namespace

// ---------------------------------------------------------------------------
juce::Image AboutComponent::createIcon (int size)
{
    juce::Image img (juce::Image::ARGB, size, size, true);
    juce::Graphics g (img);

    const float s = (float)size;
    const float r = s * 0.15f;

    // Dark navy background
    g.setColour (juce::Colour (0xff08081c));
    g.fillRoundedRectangle (0.f, 0.f, s, s, r);

    // Mini spectral waterfall — rows of bars with a gaussian peak
    const int numRows = 7;
    const int numBars = 20;
    const float margin = s * 0.09f;
    const float gridW = s - 2.f * margin;
    const float gridH = s * 0.78f;
    const float rowH = gridH / (float)numRows;
    const float barW = gridW / (float)numBars - 1.0f;

    const juce::Colour cyan (0xff00efff);
    const juce::Colour pink (0xffff26b2);

    for (int row = 0; row < numRows; ++row)
    {
        float rowAge = 1.0f - (float)row / (float)(numRows - 1);
        float yBase = margin + (float)(row + 1) * rowH;
        // Shift peak slightly per row so it looks like a moving waterfall
        float peak = 0.20f + 0.38f * (float)row / (float)numRows;

        for (int bar = 0; bar < numBars; ++bar)
        {
            float t = (float)bar / (float)(numBars - 1);
            float h = std::exp (-13.f * (t - peak) * (t - peak));
            h = std::max (0.04f, h);

            juce::Colour col = cyan.interpolatedWith (pink, t).withAlpha (0.20f + 0.80f * rowAge);
            g.setColour (col);

            float x = margin + (float)bar * (barW + 1.f);
            float barH = rowH * 0.86f * h;
            g.fillRect (x, yBase - barH, barW, barH);
        }
    }

    // Subtle cyan border
    g.setColour (juce::Colour (0x4000efff));
    g.drawRoundedRectangle (1.f, 1.f, s - 2.f, s - 2.f, r, 1.5f);

    return img;
}

// ---------------------------------------------------------------------------
AboutComponent::AboutComponent ()
{
    int rtVer = 0, drvVer = 0;
    CudaProcessor::getCudaVersions (rtVer, drvVer);
    cudaRuntimeStr = formatCudaVer (rtVer);
    cudaDriverStr = formatCudaVer (drvVer);

    icon = createIcon (88);

    addAndMakeVisible (closeButton);
    closeButton.onClick = [this]
    {
        if (auto* dw = findParentComponentOfClass<juce::DialogWindow> ())
            dw->exitModalState (0);
    };

    setSize (kWidth, kHeight);
}

// ---------------------------------------------------------------------------
void AboutComponent::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (0xff0d0d22));

    // --- Icon ---
    const int iconSz = 80;
    const int iconX = 24;
    const int iconY = 28;
    g.drawImage (icon, iconX, iconY, iconSz, iconSz, 0, 0, 88, 88);

    // --- Text block ---
    const int textX = iconX + iconSz + 20;
    const int textW = kWidth - textX - 16;
    int y = 30;

    g.setFont (juce::Font (juce::FontOptions (21.0f).withStyle ("Bold")));
    g.setColour (juce::Colour (0xff00efff));
    g.drawText ("Spectral Waterfall", textX, y, textW, 26, juce::Justification::centredLeft);
    y += 32;

    g.setFont (juce::Font (juce::FontOptions (13.0f)));
    g.setColour (juce::Colours::lightgrey);

    g.drawText ("Version " + juce::String (AppVersion::string), textX, y, textW, 18,
                juce::Justification::centredLeft);
    y += 22;

    g.drawText ("CUDA Runtime  " + cudaRuntimeStr, textX, y, textW, 18,
                juce::Justification::centredLeft);
    y += 20;

    g.drawText ("CUDA Driver   " + cudaDriverStr, textX, y, textW, 18,
                juce::Justification::centredLeft);
    y += 28;

    g.setColour (juce::Colour (0xff5555aa));
    g.drawText ("Real-time 3D spectral waterfall visualizer.", textX, y, textW, 18,
                juce::Justification::centredLeft);
    y += 20;
    g.drawText ("CUDA + OpenGL \u2022 JUCE 8 \u2022 NVIDIA Ada (SM 8.9)", textX, y, textW, 18,
                juce::Justification::centredLeft);

    // Separator above button row
    g.setColour (juce::Colour (0xff1e1e44));
    g.drawHorizontalLine (kHeight - 56, 0.f, (float)kWidth);
}

// ---------------------------------------------------------------------------
void AboutComponent::resized ()
{
    const int btnW = 100;
    const int btnH = 32;
    closeButton.setBounds ((kWidth - btnW) / 2, kHeight - 46, btnW, btnH);
}

// ---------------------------------------------------------------------------
void AboutComponent::show ()
{
    juce::DialogWindow::LaunchOptions opts;
    opts.content.setOwned (new AboutComponent ());
    opts.dialogTitle = "About Spectral Waterfall";
    opts.dialogBackgroundColour = juce::Colour (0xff0d0d22);
    opts.escapeKeyTriggersCloseButton = true;
    opts.useNativeTitleBar = false;
    opts.resizable = false;
    opts.launchAsync ();
}
