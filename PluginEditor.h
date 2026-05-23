#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "SpectralWaterfallComponent.h"

class PluginProcessor;

// =============================================================================
class PluginEditor : public juce::AudioProcessorEditor
{
  public:
    explicit PluginEditor (PluginProcessor& p);
    ~PluginEditor () override;

    // ---- Component ----------------------------------------------------------
    void paint (juce::Graphics&) override;
    void resized () override;
    void parentHierarchyChanged () override; // sets window icon when peer is ready

    // ---- Accessor for PluginProcessor ---------------------------------------
    SpectralWaterfallComponent& getWaterfall () { return waterfall; }

  private:
    PluginProcessor& processor;
    SpectralWaterfallComponent waterfall;

    // ---- Toolbar ------------------------------------------------------------
    juce::TextButton liveButton{"Live"};
    juce::TextButton fileButton{"Load File..."};
    juce::TextButton aboutButton{"?"};
    juce::FileChooser fileChooser{"Load audio file for analysis",
                                  juce::File::getSpecialLocation (juce::File::userMusicDirectory),
                                  "*.wav;*.aif;*.aiff;*.flac;*.mp3;*.ogg"};

    static constexpr int kToolbarH = 36;

    void onLiveClicked ();
    void onFileClicked ();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginEditor)
};
