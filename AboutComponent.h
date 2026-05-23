#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

// Modal "About" dialog. Show via AboutComponent::show().
// Also exposes createIcon() so the editor can set the window icon.
class AboutComponent final : public juce::Component
{
public:
    AboutComponent();

    void paint  (juce::Graphics&) override;
    void resized() override;

    // Show a non-blocking dialog window.
    static void show();

    // Creates the app icon at the requested pixel size (ARGB image).
    // Used both for the About dialog and for ComponentPeer::setIcon().
    static juce::Image createIcon (int size);

    static constexpr int kWidth  = 440;
    static constexpr int kHeight = 280;

private:
    juce::TextButton closeButton { "Close" };
    juce::Image      icon;
    juce::String     cudaRuntimeStr;
    juce::String     cudaDriverStr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AboutComponent)
};
