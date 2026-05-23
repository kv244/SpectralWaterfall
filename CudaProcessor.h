// =============================================================================
//  CudaProcessor.h
//  ---------------------------------------------------------------------------
//  Opaque C++ facade over the CUDA / cuFFT / cudaGL-interop pipeline that
//  feeds a 3D spectral waterfall visualizer.
//
//  This header is deliberately free of JUCE and OpenGL includes so it can
//  be pulled into both the JUCE .cpp side and the NVCC-compiled .cu side
//  without dragging incompatible headers across the boundary. All CUDA state
//  is hidden behind a forward-declared Impl struct (Pimpl idiom) defined
//  inside CudaProcessor.cu.
//
//  Threading contract:
//    - initialize()        : OpenGL render thread, GL context current.
//    - shutdown()          : OpenGL render thread, GL context current.
//    - pushAudioBlock()    : audio (real-time) thread. Lock-free. No alloc.
//    - renderLiveFrame()   : OpenGL render thread, GL context current.
//    - processStaticFile() : OpenGL render thread, GL context current.
//
//  Target HW: NVIDIA RTX 4070 Laptop (Ada, SM 8.9). Code is generic CUDA;
//  occupancy-tuning (block sizes) is set conservatively for portability.
// =============================================================================

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

class CudaProcessor
{
  public:
    // ---- Configuration constants -------------------------------------------
    // These are mirrored on the .cu side; if you change them, change both.
    static constexpr int FFT_SIZE = 2048; // power of two; cuFFT-friendly
    static constexpr int HOP_SIZE = 512; // 75 % overlap

    // !!! NUM_OUTPUT_BINS must remain a power of two (or at minimum a
    //     multiple of 32 for warp width and 64 for half-warp pairs).
    //
    //     Each mesh row is laid out contiguously in the VBO as
    //         NUM_OUTPUT_BINS * sizeof(float) = 512 * 4 = 2048 bytes
    //     i.e. a multiple of 128 bits (16 B) and 256 bits (32 B). That
    //     alignment is what lets `spectrumToVboRowLive` issue fully
    //     coalesced 128-bit stores straight into the mapped VBO: each
    //     warp of 32 threads writes 32 consecutive floats (128 B), one
    //     transaction per warp, no scatter, no replay.
    //
    //     If you change this value to a non-power-of-two (e.g. 500, 768
    //     without 128-B alignment), the kernel will still work but will
    //     fall into uncoalesced access patterns — expect a 3-5x drop in
    //     VBO write throughput and visible scrolling stutter under load.
    //     At that point you'd need cudaMallocPitch + a pitched VBO layout
    //     with row strides padded up to the alignment boundary, which the
    //     current interop path is not built for.
    static constexpr int NUM_OUTPUT_BINS = 512; // X-resolution of waterfall
    static_assert ((NUM_OUTPUT_BINS & (NUM_OUTPUT_BINS - 1)) == 0 && NUM_OUTPUT_BINS >= 128,
                   "NUM_OUTPUT_BINS must be a power-of-two >= 128 for coalesced VBO writes");
    static constexpr int NUM_HISTORY_ROWS = 256; // Z-resolution (live mode)
    static constexpr float MIN_FREQ_HZ = 20.0f;
    static constexpr float MAX_FREQ_HZ = 20000.0f;
    static constexpr float MIN_DB = -90.0f;
    static constexpr float MAX_DB = 0.0f;

    // PCM ring on the device: enough headroom for FFT_SIZE + drift.
    static constexpr int PCM_RING_SIZE = FFT_SIZE * 4;

    CudaProcessor ();
    ~CudaProcessor ();

    CudaProcessor (const CudaProcessor&) = delete;
    CudaProcessor& operator= (const CudaProcessor&) = delete;

    // -----------------------------------------------------------------------
    //  Lifecycle
    // -----------------------------------------------------------------------

    /** Allocate device buffers, build cuFFT plan, and register the OpenGL VBO
        for zero-copy interop. Must be called with the GL context current.

        @param glVboHandle      The GL name of the height-VBO created by
                                the JUCE component. Must already be sized to
                                NUM_HISTORY_ROWS * NUM_OUTPUT_BINS floats.
        @param sampleRate       Host sample rate.
        @returns true on success.
    */
    bool initialize (unsigned int glVboHandle, double sampleRate);

    /** Tear down all CUDA state and unregister the VBO. Idempotent. */
    void shutdown ();

    // -----------------------------------------------------------------------
    //  Live-stream path
    // -----------------------------------------------------------------------

    /** Called from the audio thread. Pushes mono samples into the lock-free
        host-side FIFO; no allocation, no syscalls.

        @returns number of samples actually accepted (may be < numSamples if
                 the GL thread is starved and the FIFO is full).
    */
    int pushAudioBlock (const float* monoSamples, int numSamples) noexcept;

    /** Called from the OpenGL render thread once per frame.

        Pops up to HOP_SIZE samples from the FIFO, advances the device PCM
        ring, runs windowing + cuFFT + log-frequency / dB conversion, and
        writes one new row into the OpenGL VBO via interop.

        @returns the new current frame index (the row that was just written),
                 which the renderer feeds to uCurrentFrameIndex. If no new
                 frame was produced this call (insufficient FIFO data), the
                 previous index is returned unchanged.
    */
    int renderLiveFrame ();

    // -----------------------------------------------------------------------
    //  Static-file path
    // -----------------------------------------------------------------------

    /** Decode the entire file into device VRAM, run a batched cuFFT across
        every hop in one shot, and populate a fresh VBO sized for the file.

        Because the static-file VBO can be much larger than the live VBO,
        this function will request a new GL buffer handle via the supplied
        allocator callback and re-register it.

        Threading: GL thread, context current.

        @param pcmHost          Pointer to host-side mono PCM (interleaved
                                channels should be pre-mixed by the caller).
        @param totalSamples     Number of samples in pcmHost.
        @param sampleRate       File's sample rate.
        @param outNumRows       Filled with numHops on success (the number
                                of rows now in the static VBO).
        @returns true on success.
    */
    bool processStaticFile (const float* pcmHost, std::size_t totalSamples, double sampleRate,
                            int& outNumRows);

    /** Re-register the OpenGL VBO with CUDA. Use after the component
        rebuilds its GL buffer (mode change, resize, etc.).

        @param glVboHandle   The new VBO name.
        @param numRows       The Z-extent of the rebuilt buffer:
                             NUM_HISTORY_ROWS for live, numHops for static. */
    bool rebindVbo (unsigned int glVboHandle, int numRows);

    // -----------------------------------------------------------------------
    //  Accessors
    // -----------------------------------------------------------------------
    int getCurrentFrameIndex () const noexcept
    {
        return currentFrameIndex.load (std::memory_order_acquire);
    }
    int getStaticNumRows () const noexcept { return staticNumRows; }
    bool isInitialized () const noexcept { return initialized; }

    /** Query CUDA runtime and driver versions (e.g. 13010 = 13.1).
        Safe to call before initialize(). Returns 0 on failure. */
    static void getCudaVersions (int& runtimeVersion, int& driverVersion) noexcept;

  private:
    // Opaque Impl. Defined in CudaProcessor.cu so this header pulls
    // zero CUDA / cuFFT / cudaGL types.
    struct Impl;
    Impl* impl = nullptr;

    std::atomic<int> currentFrameIndex{0};
    int staticNumRows = 0;
    bool initialized = false;
};
