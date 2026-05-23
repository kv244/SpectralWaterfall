#include "PluginProcessor.h"
#include "PluginEditor.h"

// =============================================================================
PluginProcessor::PluginProcessor()
    : AudioProcessor (BusesProperties()
          .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
          .withOutput ("Output", juce::AudioChannelSet::stereo(), true))
{
}

PluginProcessor::~PluginProcessor()
{
    // Editor should already be null here (JUCE deletes the editor before the
    // processor), but guard defensively in case of unusual host teardown order.
    activeEditor.store (nullptr, std::memory_order_release);
}

// =============================================================================
bool PluginProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    // Accept any combination that has at least one input channel.
    // The waterfall is an analyser; it passes audio through unchanged.
    if (layouts.getMainInputChannelSet()  == juce::AudioChannelSet::disabled() ||
        layouts.getMainOutputChannelSet() == juce::AudioChannelSet::disabled())
        return false;

    // Outputs must match inputs so we can do a transparent pass-through.
    return layouts.getMainOutputChannelSet() == layouts.getMainInputChannelSet();
}

// =============================================================================
void PluginProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // Pre-allocate the mono scratch on the message thread (prepareToPlay is
    // not called on the RT thread) so processBlock never allocates.
    monoScratch.assign (static_cast<std::size_t> (samplesPerBlock), 0.0f);

    // Forward sample rate to the waterfall if the editor already exists.
    // If it doesn't exist yet, PluginEditor's constructor will call
    // setSampleRate itself after reading back getTotalNumInputChannels().
    if (auto* e = activeEditor.load (std::memory_order_acquire))
        e->getWaterfall().setSampleRate (sampleRate);
}

void PluginProcessor::releaseResources()
{
    monoScratch.clear();
}

// =============================================================================
void PluginProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                    juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    // ---- Transparent pass-through -------------------------------------------
    // This plugin is a pure analyser; it does not modify the audio signal.
    // (Remove these lines if you want a silent / zero-latency analyser.)
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();
    for (int i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    // ---- Push mono mix to the waterfall -------------------------------------
    auto* editor = activeEditor.load (std::memory_order_acquire);
    if (editor == nullptr)
        return;

    const int numSamples = buffer.getNumSamples();
    const int numChans = totalNumInputChannels;

    // Safety: samplesPerBlock can exceed what prepareToPlay allocated if the
    // host is misbehaving. Guard without allocating; just truncate.
    const int n = juce::jmin (numSamples, (int) monoScratch.size());

    // Mix all input channels into monoScratch (stack-local, no alloc).
    std::fill (monoScratch.begin(), monoScratch.begin() + n, 0.0f);

    const float inv = (numChans > 0) ? (1.0f / (float) numChans) : 1.0f;
    for (int c = 0; c < numChans; ++c)
    {
        const float* src = buffer.getReadPointer (c);
        for (int i = 0; i < n; ++i)
            monoScratch[i] += src[i] * inv;
    }

    editor->getWaterfall().pushAudioBlock (monoScratch.data(), n);
}

// =============================================================================
juce::AudioProcessorEditor* PluginProcessor::createEditor()
{
    auto* e = new PluginEditor (*this);
    registerEditor (e);
    return e;
}

// =============================================================================
void PluginProcessor::registerEditor (PluginEditor* e)
{
    activeEditor.store (e, std::memory_order_release);
}

void PluginProcessor::unregisterEditor (PluginEditor* e)
{
    // Only clear if it is still us — guards against a race where a second
    // editor was created before the first was fully destroyed (unusual, but
    // some hosts do this during plugin scanning).
    PluginEditor* expected = e;
    activeEditor.compare_exchange_strong (expected, nullptr,
                                          std::memory_order_acq_rel);
}

// =============================================================================
// This macro tells JUCE's plugin wrapper how to instantiate the processor.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new PluginProcessor();
}
