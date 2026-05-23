#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "SpectralWaterfallComponent.h"

class PluginProcessor;

// =============================================================================
class PluginEditor : public juce::AudioProcessorEditor, public juce::MenuBarModel
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

    // ---- juce::MenuBarModel -------------------------------------------------
    juce::StringArray getMenuBarNames () override;
    juce::PopupMenu getMenuForIndex (int menuIndex, const juce::String& menuName) override;
    void menuItemSelected (int menuItemID, int topLevelMenuIndex) override;

  private:
    PluginProcessor& processor;
    SpectralWaterfallComponent waterfall;

    // ---- Menu bar -----------------------------------------------------------
    juce::MenuBarComponent menuBar;

    // ---- Mode-switch toolbar ------------------------------------------------
    juce::TextButton liveButton{"Live"};
    juce::TextButton fileButton{"Load File..."};
    juce::FileChooser fileChooser{"Load audio file for analysis",
                                  juce::File::getSpecialLocation (juce::File::userMusicDirectory),
                                  "*.wav;*.aif;*.aiff;*.flac;*.mp3;*.ogg"};

    static constexpr int kMenuBarH = 24;
    static constexpr int kToolbarH = 36;

    enum MenuItemIDs
    {
        aboutItem = 1
    };

    void onLiveClicked ();
    void onFileClicked ();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginEditor)
};
