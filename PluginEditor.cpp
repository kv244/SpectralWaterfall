#include "PluginEditor.h"
#include "PluginProcessor.h"
#include "AboutComponent.h"

// =============================================================================
PluginEditor::PluginEditor (PluginProcessor& p) : AudioProcessorEditor (&p), processor (p)
{
    // ---- Waterfall ----------------------------------------------------------
    addAndMakeVisible (waterfall);

    // ---- Toolbar buttons ----------------------------------------------------
    addAndMakeVisible (liveButton);
    addAndMakeVisible (fileButton);
    addAndMakeVisible (aboutButton);

    liveButton.setClickingTogglesState (false);
    fileButton.setClickingTogglesState (false);
    aboutButton.setClickingTogglesState (false);

    liveButton.onClick = [this] { onLiveClicked (); };
    fileButton.onClick = [this] { onFileClicked (); };
    aboutButton.onClick = [] { AboutComponent::show (); };

    // ---- Initial size -------------------------------------------------------
    setSize (1024, 512 + kToolbarH);
    setResizable (true, true);
    setResizeLimits (512, 256 + kToolbarH, 2560, 1440 + kToolbarH);

    // ---- Forward sample rate ------------------------------------------------
    const double sr = p.getSampleRate ();
    if (sr > 0.0)
        waterfall.setSampleRate (sr);
}

PluginEditor::~PluginEditor ()
{
    processor.unregisterEditor (this);
}

// =============================================================================
void PluginEditor::paint (juce::Graphics& g)
{
    // The waterfall owns its own OpenGL surface and fills its bounds entirely.
    // We only need to paint the toolbar strip.
    g.fillAll (juce::Colour (0xff0a0a14)); // near-black navy, matches fog colour
}

void PluginEditor::resized ()
{
    auto area = getLocalBounds ();

    auto toolbar = area.removeFromTop (kToolbarH);
    liveButton.setBounds (toolbar.removeFromLeft (80).reduced (4));
    fileButton.setBounds (toolbar.removeFromLeft (120).reduced (4));
    aboutButton.setBounds (toolbar.removeFromRight (36).reduced (4));

    waterfall.setBounds (area);
}

// ---------------------------------------------------------------------------
void PluginEditor::parentHierarchyChanged ()
{
    AudioProcessorEditor::parentHierarchyChanged ();
    // Set the window icon once the component has a native peer.
    // Works in standalone mode; most plugin hosts ignore setIcon().
    if (auto* peer = getPeer ())
        peer->setIcon (AboutComponent::createIcon (256));
}

// =============================================================================
void PluginEditor::onLiveClicked ()
{
    waterfall.switchToLiveStream ();
    liveButton.setEnabled (false);
    fileButton.setEnabled (true);
}

void PluginEditor::onFileClicked ()
{
    // Launch async so we never block the message thread waiting on the OS
    // file dialog — JUCE requires the async overload in plugin contexts.
    fileChooser.launchAsync (juce::FileBrowserComponent::openMode |
                                 juce::FileBrowserComponent::canSelectFiles,
                             [this] (const juce::FileChooser& fc)
                             {
                                 auto result = fc.getResult ();
                                 if (result == juce::File{})
                                     return; // user cancelled

                                 waterfall.loadStaticFile (result);
                                 liveButton.setEnabled (true);
                                 fileButton.setEnabled (false);
                             });
}
