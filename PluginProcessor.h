#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

// Forward-declared so PluginProcessor.h stays free of UI headers.
// The editor is included in PluginProcessor.cpp only.
class PluginEditor;

// =============================================================================
class PluginProcessor : public juce::AudioProcessor
{
  public:
    PluginProcessor ();
    ~PluginProcessor () override;

    // ---- AudioProcessor interface -------------------------------------------
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources () override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor () override;
    bool hasEditor () const override { return true; }

    const juce::String getName () const override { return "SpectralWaterfall"; }

    bool acceptsMidi () const override { return false; }
    bool producesMidi () const override { return false; }
    bool isMidiEffect () const override { return false; }
    double getTailLengthSeconds () const override { return 0.0; }

    int getNumPrograms () override { return 1; }
    int getCurrentProgram () override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock&) override {}
    void setStateInformation (const void*, int) override {}

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    // ---- Called by PluginEditor on construction / destruction ---------------
    //
    //  The processor needs to push audio into the waterfall component owned
    //  by the editor. Rather than giving the processor a hard dependency on
    //  the editor class, we expose a minimal interface: the editor registers
    //  itself atomically so processBlock can call it safely even on the RT
    //  thread while the editor is being constructed or torn down on the
    //  message thread.
    void registerEditor (PluginEditor* e);
    void unregisterEditor (PluginEditor* e);

  private:
    // Non-owning, atomic. Written only on the message thread (JUCE guarantees
    // createEditor / deleteEditor are message-thread-only). Read on the RT
    // thread in processBlock with acquire ordering.
    std::atomic<PluginEditor*> activeEditor{nullptr};

    // Scratch mono buffer allocated in prepareToPlay to avoid heap allocation
    // on the RT thread. processBlock mixes channels into this before handing
    // samples to the waterfall.
    std::vector<float> monoScratch;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginProcessor)
};
