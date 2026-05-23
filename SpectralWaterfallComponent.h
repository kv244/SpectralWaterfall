// =============================================================================
//  SpectralWaterfallComponent.h
//  ---------------------------------------------------------------------------
//  JUCE 8 Component + OpenGLRenderer that renders a 3D spectral waterfall.
//
//  Two operational modes:
//    - LiveStream : rolling waterfall driven by audio pushed via
//                   pushAudioBlock(). Updates one row per ~HOP_SIZE samples.
//    - StaticFile : entire file FFT'd in one batched cuFFT and rendered as
//                   a static 3D terrain.
//
//  Public API contract:
//    * Host plugin's processBlock() should mix-down to mono and call
//      pushAudioBlock(). It is real-time-safe.
//    * Host plugin's prepareToPlay() should call setSampleRate().
//    * UI thread calls loadStaticFile()  / setMode() to switch modes.
//
//  All CUDA / OpenGL work runs on JUCE's OpenGL render thread, which is
//  the "background thread" relative to the audio thread.
// =============================================================================

#pragma once

#include <juce_core/juce_core.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_opengl/juce_opengl.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>

#include <atomic>
#include <memory>
#include <mutex>

#include "CudaProcessor.h"

class SpectralWaterfallComponent : public juce::Component,
                                   public juce::OpenGLRenderer
{
public:
    enum class Mode { LiveStream, StaticFile };

    SpectralWaterfallComponent();
    ~SpectralWaterfallComponent() override;

    // ---- Host plugin integration ------------------------------------------

    /** Call from prepareToPlay() to pin the sample rate before audio starts. */
    void setSampleRate (double newSampleRate);

    /** Real-time safe. Call from processBlock() with a mono mix-down of
        the current audio block. Returns samples accepted (may be lower than
        numSamples under FIFO back-pressure; this is fine — we just drop). */
    int pushAudioBlock (const float* monoSamples, int numSamples) noexcept;

    /** UI thread. Decode an audio file and switch to StaticFile mode. */
    bool loadStaticFile (const juce::File& file);

    /** UI thread. Switch back to LiveStream mode. */
    void switchToLiveStream();

    Mode getMode() const noexcept { return mode.load (std::memory_order_acquire); }

    // ---- juce::Component ---------------------------------------------------
    void paint    (juce::Graphics&) override {}   // disabled; GL paints
    void resized() override;

    // ---- juce::OpenGLRenderer ---------------------------------------------
    void newOpenGLContextCreated() override;
    void renderOpenGL()            override;
    void openGLContextClosing()    override;

private:
    // ---- GL helpers --------------------------------------------------------
    bool buildShaders();
    bool buildLiveGeometry();                          // height-VBO + index EBO
    bool buildStaticGeometry (int numRows);            // resized for the file
    void releaseGeometry();
    void setCameraUniforms();

    // ---- State -------------------------------------------------------------
    juce::OpenGLContext   openGLContext;
    std::unique_ptr<juce::OpenGLShaderProgram> shader;

    // Cached uniform handles (re-located after each shader rebuild)
    std::unique_ptr<juce::OpenGLShaderProgram::Uniform> uMVP;
    std::unique_ptr<juce::OpenGLShaderProgram::Uniform> uCurrentFrameIndex;
    std::unique_ptr<juce::OpenGLShaderProgram::Uniform> uNumFreqBins;
    std::unique_ptr<juce::OpenGLShaderProgram::Uniform> uNumRows;
    std::unique_ptr<juce::OpenGLShaderProgram::Uniform> uHeightScale;
    std::unique_ptr<juce::OpenGLShaderProgram::Uniform> uTime;

    GLuint vao = 0;
    GLuint vboHeights = 0;
    GLuint ebo = 0;
    int    indexCount = 0;
    int    geometryNumRows = 0;          // either NUM_HISTORY_ROWS or numHops

    // CUDA backend
    CudaProcessor cuda;

    // Mode flip-flop, set from UI, read from GL thread.
    std::atomic<Mode>  mode { Mode::LiveStream };
    std::atomic<bool>  pendingStaticUpload { false };
    std::atomic<bool>  pendingLiveRebuild  { false };

    // Static-file data, prepared on UI thread, consumed on GL thread.
    std::mutex                  staticDataMutex;
    juce::AudioBuffer<float>    staticPcmMono;
    double                      staticSampleRate = 44100.0;

    // Sample rate (live)
    std::atomic<double> sampleRate { 44100.0 };

    // Camera
    juce::Time           startTime { juce::Time::getCurrentTime() };
    float                cameraOrbit = 0.0f;

    // Audio-format support for loadStaticFile().
    juce::AudioFormatManager audioFormatManager;

    // Disallow copying.
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SpectralWaterfallComponent)
};
