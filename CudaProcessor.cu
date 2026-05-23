// =============================================================================
//  CudaProcessor.cu
//  ---------------------------------------------------------------------------
//  CUDA + cuFFT + cudaGL-interop implementation behind CudaProcessor.h.
//
//  Build:
//     nvcc -std=c++17 -O3 -arch=sm_89 -c CudaProcessor.cu
//     # sm_89 = Ada / RTX 4070 Laptop. Use sm_86 for Ampere fallback.
//  Link against:
//     cudart, cufft, opengl32 (Win) / GL (Linux) / OpenGL.framework (mac)
//
//  Conventions:
//    - All kernel launches use a CUDA stream owned by Impl (never the default
//      stream, so we never serialize against unrelated work).
//    - All device pointers are owned by Impl and freed in shutdown(). The
//      destructor is idempotent.
//    - The host audio FIFO is a Single-Producer/Single-Consumer ring buffer
//      with two atomic indices; the audio thread is producer, the GL thread
//      is consumer. No malloc / free / locks on either path after init.
//    - Errors during real-time paths set an error flag but never throw or
//      abort; the visualizer should degrade gracefully (frame goes blank).
// =============================================================================

#include "CudaProcessor.h"

// Some CUDA distributions expect a GL header before cuda_gl_interop.h on
// certain platforms. Guard for portability.
#if defined(_WIN32)
#include <windows.h>
#include <GL/gl.h>
#elif defined(__APPLE__)
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cufft.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

// -----------------------------------------------------------------------------
//  Error helpers
// -----------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                                           \
    do                                                                                             \
    {                                                                                              \
        cudaError_t _e = (expr);                                                                   \
        if (_e != cudaSuccess)                                                                     \
        {                                                                                          \
            std::fprintf (stderr, "[CUDA] %s @ %s:%d : %s\n", #expr, __FILE__, __LINE__,           \
                          cudaGetErrorString (_e));                                                \
            return false;                                                                          \
        }                                                                                          \
    } while (0)

#define CUFFT_CHECK(expr)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cufftResult _r = (expr);                                                                   \
        if (_r != CUFFT_SUCCESS)                                                                   \
        {                                                                                          \
            std::fprintf (stderr, "[cuFFT] %s @ %s:%d : %d\n", #expr, __FILE__, __LINE__,          \
                          int (_r));                                                               \
            return false;                                                                          \
        }                                                                                          \
    } while (0)

// Non-returning soft check, for the real-time render path.
#define CUDA_SOFT(expr)                                                                            \
    do                                                                                             \
    {                                                                                              \
        cudaError_t _e = (expr);                                                                   \
        if (_e != cudaSuccess)                                                                     \
        {                                                                                          \
            std::fprintf (stderr, "[CUDA-soft] %s : %s\n", #expr, cudaGetErrorString (_e));        \
        }                                                                                          \
    } while (0)

// Non-returning soft check for cuFFT — use on the render path where we must
// not skip cleanup (e.g. VBO unmap) on FFT error.
#define CUFFT_SOFT(expr)                                                                           \
    do                                                                                             \
    {                                                                                              \
        cufftResult _r = (expr);                                                                   \
        if (_r != CUFFT_SUCCESS)                                                                   \
        {                                                                                          \
            std::fprintf (stderr, "[cuFFT-soft] %s : %d\n", #expr, int (_r));                      \
        }                                                                                          \
    } while (0)

// -----------------------------------------------------------------------------
//  Single-Producer / Single-Consumer host FIFO
//  ---------------------------------------------------------------------------
//  We deliberately roll our own (rather than use juce::AbstractFifo) here
//  because this translation unit is compiled by NVCC and must not depend on
//  JUCE. The semantics are equivalent: lock-free, allocation-free after init,
//  and safe between one producer thread and one consumer thread.
// -----------------------------------------------------------------------------
struct SpscFifo
{
    // Size is fixed at construction. Must be power of two for fast wrap.
    int capacity = 0;
    std::vector<float> buffer;
    std::atomic<int> writeIdx{0}; // monotonically increasing
    std::atomic<int> readIdx{0};

    void init (int capacityPow2)
    {
        // round to next power of two
        int n = 1;
        while (n < capacityPow2)
            n <<= 1;
        capacity = n;
        buffer.assign (static_cast<std::size_t> (capacity), 0.0f);
        writeIdx.store (0);
        readIdx.store (0);
    }

    int mask () const noexcept { return capacity - 1; }
    int used () const noexcept
    {
        return writeIdx.load (std::memory_order_acquire) - readIdx.load (std::memory_order_acquire);
    }
    int free () const noexcept { return capacity - used (); }

    // Producer side. Audio thread.
    int push (const float* src, int n) noexcept
    {
        int avail = free ();
        int toWrite = n < avail ? n : avail;
        int w = writeIdx.load (std::memory_order_relaxed);
        for (int i = 0; i < toWrite; ++i)
            buffer[(w + i) & mask ()] = src[i];
        writeIdx.store (w + toWrite, std::memory_order_release);
        return toWrite;
    }

    // Consumer side. GL thread.
    int pop (float* dst, int n) noexcept
    {
        int avail = used ();
        int toRead = n < avail ? n : avail;
        int r = readIdx.load (std::memory_order_relaxed);
        for (int i = 0; i < toRead; ++i)
            dst[i] = buffer[(r + i) & mask ()];
        readIdx.store (r + toRead, std::memory_order_release);
        return toRead;
    }
};

// =============================================================================
//  CUDA kernels
// =============================================================================

// ---------------------------------------------------------------------------
//  applyHannWindowFromRing
//      Reads FFT_SIZE samples ending at writeIdx from a circular PCM ring on
//      the device, multiplies by a precomputed Hann window, and writes a
//      contiguous FFT_SIZE-length real signal ready for cuFFT R2C.
// ---------------------------------------------------------------------------
__global__ void applyHannWindowFromRing (const float* __restrict__ pcmRing,
                                         int writeIdx, // one past last written sample
                                         int ringSize, const float* __restrict__ hannLUT,
                                         float* __restrict__ windowedOut, int fftSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fftSize)
        return;

    // Sample index in the ring corresponding to position i in the window.
    // The most recent sample sits at writeIdx-1; the window extends back
    // fftSize samples.
    // C's `%` operator may yield a negative remainder when the dividend is
    // negative (writeIdx - fftSize can underflow once writeIdx < fftSize on
    // the first few frames after init). A branchful `if` here has been seen
    // to produce inconsistent results across nvcc release builds with
    // -O3 PTX scheduling. Express the wrap as an explicit ternary so the
    // compiler emits a predicated select (sel.s32) instead of a branch —
    // this guarantees srcIdx is always in [0, ringSize) before the load.
    int srcIdx = (writeIdx - fftSize + i) % ringSize;
    srcIdx = (srcIdx < 0) ? (srcIdx + ringSize) : srcIdx;

    windowedOut[i] = pcmRing[srcIdx] * hannLUT[i];
}

// ---------------------------------------------------------------------------
//  applyHannWindowBatched
//      Static-file pre-FFT staging: for each hop h in [0, numHops), produce a
//      contiguous FFT_SIZE block of windowed samples in the batched layout
//      cuFFT expects (hop-major: out[h*fftSize + i]).
//
//      Launched as a 2D grid:
//          gridDim.y = numHops
//          gridDim.x = ceil(fftSize / blockDim.x)
// ---------------------------------------------------------------------------
__global__ void applyHannWindowBatched (const float* __restrict__ pcm, std::size_t totalSamples,
                                        const float* __restrict__ hannLUT,
                                        float* __restrict__ windowedOut, int fftSize, int hopSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int hop = blockIdx.y;
    if (i >= fftSize)
        return;

    std::size_t srcIdx = (std::size_t)hop * (std::size_t)hopSize + (std::size_t)i;
    float sample = (srcIdx < totalSamples) ? pcm[srcIdx] : 0.0f;

    windowedOut[(std::size_t)hop * fftSize + i] = sample * hannLUT[i];
}

// ---------------------------------------------------------------------------
//  spectrumToVboRowLive
//      Takes the FFT_SIZE/2+1 complex spectrum bins from one cuFFT R2C and
//      writes one row of the VBO. For each output bin (log-spaced between
//      MIN_FREQ_HZ and MAX_FREQ_HZ) we interpolate between the two nearest
//      linear FFT bins, take magnitude, convert to dB, and normalize to
//      [0, 1].
//
//      vbo points at the *whole* mapped VBO; we offset by rowSlot.
// ---------------------------------------------------------------------------
__global__ void spectrumToVboRowLive (const cufftComplex* __restrict__ spectrum, int fftSize,
                                      float sampleRate, float* __restrict__ vbo, int rowSlot,
                                      int numOutputBins, float minFreq, float maxFreq, float minDb,
                                      float maxDb)
{
    int outBin = blockIdx.x * blockDim.x + threadIdx.x;
    if (outBin >= numOutputBins)
        return;

    // Log frequency for this output column.
    float t = float (outBin) / float (numOutputBins - 1);
    float freq = minFreq * powf (maxFreq / minFreq, t);
    float fftBin = freq * float (fftSize) / sampleRate;

    int maxIdx = fftSize / 2;
    int b0 = (int)floorf (fftBin);
    int b1 = b0 + 1;
    float frac = fftBin - (float)b0;
    if (b0 < 0)
    {
        b0 = 0;
        b1 = 0;
        frac = 0.f;
    }
    if (b1 > maxIdx)
    {
        b1 = maxIdx;
    }
    if (b0 > maxIdx)
    {
        b0 = maxIdx;
    }

    cufftComplex c0 = spectrum[b0];
    cufftComplex c1 = spectrum[b1];

    float mag0 = sqrtf (c0.x * c0.x + c0.y * c0.y);
    float mag1 = sqrtf (c1.x * c1.x + c1.y * c1.y);
    float mag = mag0 * (1.0f - frac) + mag1 * frac;

    // Scale: cuFFT R2C returns un-normalised; 2/N gives single-sided amp.
    mag *= (2.0f / float (fftSize));

    float db = 20.0f * log10f (fmaxf (mag, 1e-10f));
    float n = (db - minDb) / (maxDb - minDb);
    n = fminf (fmaxf (n, 0.0f), 1.0f);

    vbo[(std::size_t)rowSlot * numOutputBins + outBin] = n;
}

// ---------------------------------------------------------------------------
//  spectrumToVboBatched
//      Static-file post-FFT: 2D grid. y indexes hops, x indexes output bins.
//      One kernel call populates the entire static VBO.
// ---------------------------------------------------------------------------
__global__ void spectrumToVboBatched (const cufftComplex* __restrict__ spectra, int fftSize,
                                      float sampleRate, float* __restrict__ vbo, int numOutputBins,
                                      int numHops, float minFreq, float maxFreq, float minDb,
                                      float maxDb)
{
    int outBin = blockIdx.x * blockDim.x + threadIdx.x;
    int hop = blockIdx.y;
    if (outBin >= numOutputBins || hop >= numHops)
        return;

    const cufftComplex* spectrum = spectra + (std::size_t)hop * (fftSize / 2 + 1);

    float t = float (outBin) / float (numOutputBins - 1);
    float freq = minFreq * powf (maxFreq / minFreq, t);
    float fftBin = freq * float (fftSize) / sampleRate;

    int maxIdx = fftSize / 2;
    int b0 = (int)floorf (fftBin);
    int b1 = b0 + 1;
    float frac = fftBin - (float)b0;
    if (b0 < 0)
    {
        b0 = 0;
        b1 = 0;
        frac = 0.f;
    }
    if (b1 > maxIdx)
    {
        b1 = maxIdx;
    }
    if (b0 > maxIdx)
    {
        b0 = maxIdx;
    }

    cufftComplex c0 = spectrum[b0];
    cufftComplex c1 = spectrum[b1];

    float mag0 = sqrtf (c0.x * c0.x + c0.y * c0.y);
    float mag1 = sqrtf (c1.x * c1.x + c1.y * c1.y);
    float mag = mag0 * (1.0f - frac) + mag1 * frac;

    mag *= (2.0f / float (fftSize));

    float db = 20.0f * log10f (fmaxf (mag, 1e-10f));
    float n = (db - minDb) / (maxDb - minDb);
    n = fminf (fmaxf (n, 0.0f), 1.0f);

    vbo[(std::size_t)hop * numOutputBins + outBin] = n;
}

// =============================================================================
//  Pimpl: every device-side resource lives here.
// =============================================================================
struct CudaProcessor::Impl
{
    cudaStream_t stream = nullptr;
    cufftHandle liveFftPlan = 0; // single-batch R2C
    cufftHandle batchPlan = 0; // many-batch R2C for static
    bool batchPlanValid = false;

    // Device buffers (live)
    float* dPcmRing = nullptr; // PCM_RING_SIZE floats
    float* dHannLUT = nullptr; // FFT_SIZE floats
    float* dWindowed = nullptr; // FFT_SIZE floats
    cufftComplex* dSpectrum = nullptr; // FFT_SIZE/2+1 cufftComplex

    // Device buffers (static)
    float* dStaticPcm = nullptr; // full file
    float* dStaticWindow = nullptr; // numHops * FFT_SIZE
    cufftComplex* dStaticSpectra = nullptr; // numHops * (FFT_SIZE/2+1)

    // CUDA-GL interop
    cudaGraphicsResource* cudaVboRes = nullptr; // currently registered VBO
    unsigned int registeredVbo = 0;

    // Host-side staging for FIFO->GPU
    float* hStagingPinned = nullptr; // pinned host memory, HOP_SIZE floats

    // FIFO between audio thread and GL thread.
    SpscFifo fifo;

    // Ring-buffer write head (in samples, monotonically increasing modulo).
    int pcmWriteIdx = 0;

    // Cached state
    double sampleRate = 44100.0;
    int liveRowSlot = 0; // round-robin row in [0, NUM_HISTORY_ROWS)

    // Sub-helpers ------------------------------------------------------------

    bool allocLiveBuffers ()
    {
        CUDA_CHECK (cudaMalloc (&dPcmRing, sizeof (float) * PCM_RING_SIZE));
        CUDA_CHECK (cudaMemsetAsync (dPcmRing, 0, sizeof (float) * PCM_RING_SIZE, stream));
        CUDA_CHECK (cudaMalloc (&dHannLUT, sizeof (float) * FFT_SIZE));
        CUDA_CHECK (cudaMalloc (&dWindowed, sizeof (float) * FFT_SIZE));
        CUDA_CHECK (cudaMalloc (&dSpectrum, sizeof (cufftComplex) * (FFT_SIZE / 2 + 1)));

        // Build Hann window on host, copy to device.
        std::vector<float> hann (FFT_SIZE);
        constexpr double TWO_PI = 6.283185307179586476925286766559;
        for (int i = 0; i < FFT_SIZE; ++i)
            hann[(std::size_t)i] = 0.5f * (1.0f - std::cos ((float)(TWO_PI * i / (FFT_SIZE - 1))));
        CUDA_CHECK (cudaMemcpyAsync (dHannLUT, hann.data (), sizeof (float) * FFT_SIZE,
                                     cudaMemcpyHostToDevice, stream));
        return true;
    }

    void freeLiveBuffers () noexcept
    {
        if (dPcmRing)
        {
            cudaFree (dPcmRing);
            dPcmRing = nullptr;
        }
        if (dHannLUT)
        {
            cudaFree (dHannLUT);
            dHannLUT = nullptr;
        }
        if (dWindowed)
        {
            cudaFree (dWindowed);
            dWindowed = nullptr;
        }
        if (dSpectrum)
        {
            cudaFree (dSpectrum);
            dSpectrum = nullptr;
        }
    }

    void freeStaticBuffers () noexcept
    {
        if (dStaticPcm)
        {
            cudaFree (dStaticPcm);
            dStaticPcm = nullptr;
        }
        if (dStaticWindow)
        {
            cudaFree (dStaticWindow);
            dStaticWindow = nullptr;
        }
        if (dStaticSpectra)
        {
            cudaFree (dStaticSpectra);
            dStaticSpectra = nullptr;
        }
        if (batchPlanValid)
        {
            cufftDestroy (batchPlan);
            batchPlan = 0;
            batchPlanValid = false;
        }
    }

    void unregisterVbo () noexcept
    {
        if (cudaVboRes != nullptr)
        {
            cudaGraphicsUnregisterResource (cudaVboRes);
            cudaVboRes = nullptr;
            registeredVbo = 0;
        }
    }
};

// =============================================================================
//  Public-facing CudaProcessor implementations
// =============================================================================

void CudaProcessor::getCudaVersions (int& runtimeVersion, int& driverVersion) noexcept
{
    runtimeVersion = 0;
    driverVersion = 0;
    cudaRuntimeGetVersion (&runtimeVersion);
    cudaDriverGetVersion (&driverVersion);
}

CudaProcessor::CudaProcessor () = default;

CudaProcessor::~CudaProcessor ()
{
    shutdown ();
}

// -----------------------------------------------------------------------------
bool CudaProcessor::initialize (unsigned int glVboHandle, double sampleRate)
{
    if (initialized)
        return true;

    impl = new (std::nothrow) Impl ();
    if (impl == nullptr)
        return false;

    impl->sampleRate = sampleRate;

    // On any failure after impl is created, shutdown() cleans up all partially
    // initialised resources via null-checks — so always go through it rather
    // than leaking impl or its sub-allocations.
    auto fail = [this] (const char* msg) -> bool
    {
        std::fprintf (stderr, "[CUDA] initialize: %s\n", msg);
        shutdown ();
        return false;
    };

    int deviceCount = 0;
    if (cudaGetDeviceCount (&deviceCount) != cudaSuccess || deviceCount == 0)
        return fail ("no CUDA-capable device found");

    if (cudaSetDevice (0) != cudaSuccess)
        return fail ("cudaSetDevice(0) failed");

    // Dedicated stream so we never serialize against unrelated CUDA work.
    if (cudaStreamCreateWithFlags (&impl->stream, cudaStreamNonBlocking) != cudaSuccess)
        return fail ("cudaStreamCreateWithFlags failed");

    // Pinned staging for the audio-thread -> GL-thread handoff.
    if (cudaHostAlloc ((void**)&impl->hStagingPinned, sizeof (float) * HOP_SIZE,
                       cudaHostAllocDefault) != cudaSuccess)
        return fail ("cudaHostAlloc for staging buffer failed");

    if (!impl->allocLiveBuffers ())
        return fail ("allocLiveBuffers failed");

    // FIFO: pick a size that comfortably absorbs jitter at 60 fps even with
    // chunky 4096-sample audio blocks.
    impl->fifo.init (16384);

    // Build the cuFFT plan for the per-frame live FFT.
    if (cufftPlan1d (&impl->liveFftPlan, FFT_SIZE, CUFFT_R2C, 1) != CUFFT_SUCCESS)
        return fail ("cufftPlan1d failed");

    if (cufftSetStream (impl->liveFftPlan, impl->stream) != CUFFT_SUCCESS)
        return fail ("cufftSetStream (live plan) failed");

    // Register the VBO for zero-copy interop.
    if (cudaGraphicsGLRegisterBuffer (&impl->cudaVboRes, (GLuint)glVboHandle,
                                      cudaGraphicsMapFlagsWriteDiscard) != cudaSuccess)
        return fail ("cudaGraphicsGLRegisterBuffer failed");

    impl->registeredVbo = glVboHandle;
    initialized = true;
    return true;
}

// -----------------------------------------------------------------------------
void CudaProcessor::shutdown ()
{
    // ---- Defensive impl check (MUST stay at the very top of this function).
    //
    //  shutdown() may be called from three different paths:
    //    (a) the normal destructor, after a successful initialize()
    //    (b) the destructor, after a FAILED initialize() that bailed
    //        partway through and called shutdown() itself for cleanup
    //    (c) an external caller (e.g. plugin releaseResources) before
    //        initialize() has ever run
    //
    //  In cases (b) and (c), `impl` may legitimately be nullptr. Touching
    //  impl->stream, impl->fifo, etc. in that state is undefined behaviour
    //  (typically a segfault on Linux, an access violation on Windows).
    //  The check below must precede every other line in this function. Do
    //  not move CUDA syncs, frees, or stream destruction above it.
    if (impl == nullptr)
    {
        initialized = false;
        currentFrameIndex.store (0, std::memory_order_release);
        staticNumRows = 0;
        return;
    }

    // Drain the stream before destroying anything it might be touching.
    if (impl->stream)
        cudaStreamSynchronize (impl->stream);

    impl->unregisterVbo ();

    if (impl->liveFftPlan)
    {
        cufftDestroy (impl->liveFftPlan);
        impl->liveFftPlan = 0;
    }
    impl->freeStaticBuffers ();
    impl->freeLiveBuffers ();

    if (impl->hStagingPinned)
    {
        cudaFreeHost (impl->hStagingPinned);
        impl->hStagingPinned = nullptr;
    }
    if (impl->stream)
    {
        cudaStreamDestroy (impl->stream);
        impl->stream = nullptr;
    }

    delete impl;
    impl = nullptr;
    initialized = false;
    currentFrameIndex.store (0, std::memory_order_release);
    staticNumRows = 0;
}

// -----------------------------------------------------------------------------
int CudaProcessor::pushAudioBlock (const float* monoSamples, int numSamples) noexcept
{
    // Hot path: audio thread. Must remain allocation- and lock-free.
    if (!initialized || impl == nullptr || monoSamples == nullptr || numSamples <= 0)
        return 0;
    return impl->fifo.push (monoSamples, numSamples);
}

// -----------------------------------------------------------------------------
int CudaProcessor::renderLiveFrame ()
{
    if (!initialized || impl == nullptr)
        return currentFrameIndex.load ();

    // Need at least HOP_SIZE new samples to make a new spectrogram row.
    if (impl->fifo.used () < HOP_SIZE)
        return currentFrameIndex.load ();

    // Pop HOP_SIZE samples from the FIFO into pinned host memory, then ship
    // to the device asynchronously on our stream.
    int popped = impl->fifo.pop (impl->hStagingPinned, HOP_SIZE);
    if (popped < HOP_SIZE)
    {
        // Spurious: another consumer? Shouldn't happen with SPSC contract.
        return currentFrameIndex.load ();
    }

    // Upload to a temporary device staging area inside the ring. Simpler
    // approach: use cudaMemcpyAsync directly into the ring, splitting at
    // the wrap if necessary.
    int ringSize = PCM_RING_SIZE;
    int w = impl->pcmWriteIdx;
    int firstChunk = std::min (HOP_SIZE, ringSize - w);
    int secondChunk = HOP_SIZE - firstChunk;

    CUDA_SOFT (cudaMemcpyAsync (impl->dPcmRing + w, impl->hStagingPinned,
                                sizeof (float) * firstChunk, cudaMemcpyHostToDevice, impl->stream));
    if (secondChunk > 0)
    {
        CUDA_SOFT (cudaMemcpyAsync (impl->dPcmRing, impl->hStagingPinned + firstChunk,
                                    sizeof (float) * secondChunk, cudaMemcpyHostToDevice,
                                    impl->stream));
    }

    impl->pcmWriteIdx = (impl->pcmWriteIdx + HOP_SIZE) % ringSize;

    // -- Map the VBO for CUDA write ------------------------------------------
    CUDA_SOFT (cudaGraphicsMapResources (1, &impl->cudaVboRes, impl->stream));
    float* dVbo = nullptr;
    size_t vboBytes = 0;
    CUDA_SOFT (cudaGraphicsResourceGetMappedPointer ((void**)&dVbo, &vboBytes, impl->cudaVboRes));

    if (dVbo != nullptr)
    {
        // 1. Hann window
        {
            int threads = 256;
            int blocks = (FFT_SIZE + threads - 1) / threads;
            applyHannWindowFromRing<<<blocks, threads, 0, impl->stream>>> (
                impl->dPcmRing, impl->pcmWriteIdx, ringSize, impl->dHannLUT, impl->dWindowed,
                FFT_SIZE);
        }

        // 2. cuFFT R2C (soft check: must not skip the VBO unmap below on error)
        CUFFT_SOFT (cufftExecR2C (impl->liveFftPlan, impl->dWindowed, impl->dSpectrum));

        // 3. Magnitude / log-freq / dB -> mapped VBO row.
        int rowSlot = impl->liveRowSlot;
        {
            int threads = 256;
            int blocks = (NUM_OUTPUT_BINS + threads - 1) / threads;
            spectrumToVboRowLive<<<blocks, threads, 0, impl->stream>>> (
                impl->dSpectrum, FFT_SIZE, (float)impl->sampleRate, dVbo, rowSlot, NUM_OUTPUT_BINS,
                MIN_FREQ_HZ, MAX_FREQ_HZ, MIN_DB, MAX_DB);
        }

        impl->liveRowSlot = (impl->liveRowSlot + 1) % NUM_HISTORY_ROWS;
    }

    CUDA_SOFT (cudaGraphicsUnmapResources (1, &impl->cudaVboRes, impl->stream));

    // We do *not* synchronize here. The GL draw call sees a buffer that the
    // GL driver will fence against our stream automatically via the interop
    // unmap. cudaGraphicsUnmapResources inserts the required dependency.

    // Return the slot we just wrote (one less than liveRowSlot after the
    // increment above), wrapped. This is what the shader interprets as
    // "age 0 = newest".
    int justWritten = (impl->liveRowSlot - 1 + NUM_HISTORY_ROWS) % NUM_HISTORY_ROWS;
    currentFrameIndex.store (justWritten, std::memory_order_release);
    return justWritten;
}

// -----------------------------------------------------------------------------
bool CudaProcessor::rebindVbo (unsigned int glVboHandle, int numRows)
{
    if (impl == nullptr)
        return false;

    // Drain to avoid unregistering a resource that has in-flight work.
    cudaStreamSynchronize (impl->stream);
    impl->unregisterVbo ();

    CUDA_CHECK (cudaGraphicsGLRegisterBuffer (&impl->cudaVboRes, (GLuint)glVboHandle,
                                              cudaGraphicsMapFlagsWriteDiscard));
    impl->registeredVbo = glVboHandle;
    staticNumRows = numRows;
    return true;
}

// -----------------------------------------------------------------------------
bool CudaProcessor::processStaticFile (const float* pcmHost, std::size_t totalSamples,
                                       double sampleRate, int& outNumRows)
{
    outNumRows = 0;
    if (!initialized || impl == nullptr || pcmHost == nullptr ||
        totalSamples < (std::size_t)FFT_SIZE)
        return false;

    impl->sampleRate = sampleRate;

    int numHops = (int)((totalSamples - FFT_SIZE) / HOP_SIZE) + 1;
    if (numHops <= 0)
        return false;

    // -- Free any previous static allocation ----------------------------------
    impl->freeStaticBuffers ();

    // -- Upload full PCM to VRAM ---------------------------------------------
    CUDA_CHECK (cudaMalloc (&impl->dStaticPcm, sizeof (float) * totalSamples));
    CUDA_CHECK (cudaMemcpyAsync (impl->dStaticPcm, pcmHost, sizeof (float) * totalSamples,
                                 cudaMemcpyHostToDevice, impl->stream));

    // -- Allocate batched scratch for windowed time-domain frames ------------
    std::size_t windowBytes = sizeof (float) * (std::size_t)numHops * FFT_SIZE;
    CUDA_CHECK (cudaMalloc (&impl->dStaticWindow, windowBytes));

    // -- Allocate batched output complex spectra -----------------------------
    std::size_t spectraBytes = sizeof (cufftComplex) * (std::size_t)numHops * (FFT_SIZE / 2 + 1);
    CUDA_CHECK (cudaMalloc (&impl->dStaticSpectra, spectraBytes));

    // -- Window all hops in one launch ---------------------------------------
    {
        dim3 threads (256, 1);
        dim3 blocks ((FFT_SIZE + threads.x - 1) / threads.x, (unsigned int)numHops);
        applyHannWindowBatched<<<blocks, threads, 0, impl->stream>>> (
            impl->dStaticPcm, totalSamples, impl->dHannLUT, impl->dStaticWindow, FFT_SIZE,
            HOP_SIZE);
    }

    // -- Build the batched cuFFT plan ----------------------------------------
    int n[1] = {FFT_SIZE};
    int inEmb[1] = {FFT_SIZE};
    int otEmb[1] = {FFT_SIZE / 2 + 1};
    CUFFT_CHECK (cufftPlanMany (&impl->batchPlan,
                                /*rank*/ 1, n, inEmb, /*istride*/ 1, /*idist*/ FFT_SIZE, otEmb,
                                /*ostride*/ 1, /*odist*/ FFT_SIZE / 2 + 1, CUFFT_R2C, numHops));
    impl->batchPlanValid = true;
    CUFFT_CHECK (cufftSetStream (impl->batchPlan, impl->stream));

    // -- Execute the batched FFT ---------------------------------------------
    CUFFT_CHECK (cufftExecR2C (impl->batchPlan, impl->dStaticWindow, impl->dStaticSpectra));

    // -- Map the (newly resized) VBO and write all rows ----------------------
    CUDA_CHECK (cudaGraphicsMapResources (1, &impl->cudaVboRes, impl->stream));
    float* dVbo = nullptr;
    size_t vboBytes = 0;
    CUDA_CHECK (cudaGraphicsResourceGetMappedPointer ((void**)&dVbo, &vboBytes, impl->cudaVboRes));

    std::size_t requiredBytes = sizeof (float) * (std::size_t)numHops * NUM_OUTPUT_BINS;
    if (vboBytes < requiredBytes)
    {
        std::fprintf (stderr,
                      "[CUDA] static VBO too small: have %zu, need %zu. "
                      "Call rebindVbo() with a buffer of the right size first.\n",
                      vboBytes, requiredBytes);
        cudaGraphicsUnmapResources (1, &impl->cudaVboRes, impl->stream);
        return false;
    }

    {
        dim3 threads (256, 1);
        dim3 blocks ((NUM_OUTPUT_BINS + threads.x - 1) / threads.x, (unsigned int)numHops);
        spectrumToVboBatched<<<blocks, threads, 0, impl->stream>>> (
            impl->dStaticSpectra, FFT_SIZE, (float)sampleRate, dVbo, NUM_OUTPUT_BINS, numHops,
            MIN_FREQ_HZ, MAX_FREQ_HZ, MIN_DB, MAX_DB);
    }

    CUDA_CHECK (cudaGraphicsUnmapResources (1, &impl->cudaVboRes, impl->stream));

    // Finish before returning so the renderer can immediately draw the file.
    CUDA_CHECK (cudaStreamSynchronize (impl->stream));

    staticNumRows = numHops;
    outNumRows = numHops;
    return true;
}
