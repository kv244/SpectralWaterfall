# Spectral Waterfall Visualizer

[![CI](https://github.com/kv244/SpectralWaterfall/actions/workflows/ci.yml/badge.svg)](https://github.com/kv244/SpectralWaterfall/actions/workflows/ci.yml)

Real-time 3D audio spectrum waterfall for JUCE 8, accelerated end-to-end on
CUDA with zero-copy CUDA↔OpenGL interop. Targets NVIDIA Ada (RTX 4070
Laptop, `sm_89`). Two render modes share a single shader: a rolling
60 Hz live stream from the plugin's audio thread, and a one-shot batched
FFT of an entire audio file.

---

## Architecture summary

- **`CudaProcessor.h`** is pure C++. No CUDA, no JUCE. The host plugin
  includes it freely; everything heavy lives behind a Pimpl.
- **Audio thread never touches CUDA.** It only calls
  `pushAudioBlock(monoFloat*, n)`. If the SPSC FIFO is full, samples are
  silently dropped — by design, this is the only RT-safe failure mode.
- **All allocation up-front.** `cudaMalloc` and `cufftPlanMany` happen in
  `initialize()` (live mode) or once per file in `processStaticFile()`.
  The render thread allocates nothing.
- **One shader, two modes.** Static mode pins the frame index to
  `numRows - 1` so the end of the file lands at the camera (time flows
  toward the viewer). Live mode tracks the just-written ring slot. Same
  GLSL, same Z mapping, no branching in the vertex shader.
- **One float per vertex.** The VBO stores only height; the vertex
  shader reconstructs `(x, y, z)` from `gl_VertexID`, and the fragment
  shader derives true per-pixel surface normals via screen-space
  derivatives (`dFdx`/`dFdy` of `vWorldPos`). No per-vertex normals
  stored, no neighbour samples needed, full per-pixel topology shading
  on the RTX 4070's spare fragment headroom.

## Pipeline

```
processBlock ──► SpscFifo (lock-free) ──► GL render thread (60 Hz)
                                              │
                                              ▼
                                  pop HOP_SIZE samples → pinned staging
                                              │  cudaMemcpyAsync (split for ring-wrap)
                                              ▼
                                  Hann window kernel  ┐
                                  cufftExecR2C        │  all on a
                                  log-freq + dB map   │  non-blocking stream
                                  write into mapped VBO ┘
                                              │
                                              ▼
                                  glDrawElements ── vertex shader reconstructs
                                                    (x,y,z), warps Z via
                                                    uCurrentFrameIndex
```

## Files

| File | Role |
|---|---|
| `CudaProcessor.h` | Opaque C++ facade — public API for the host plugin |
| `CudaProcessor.cu` | All CUDA: kernels, cuFFT, GL interop, SPSC FIFO |
| `SpectralWaterfallComponent.h` | `juce::Component` + `juce::OpenGLRenderer` |
| `SpectralWaterfallComponent.cpp` | GLSL shaders, geometry, camera, render loop |

## Build

```
nvcc -std=c++17 -O3 -arch=sm_89 -Xcompiler -fPIC -c CudaProcessor.cu
```

Link against `cudart`, `cufft`, and your platform's GL loader. JUCE's
`juce_opengl` is fine on the C++ side; the `.cu` side has platform-guarded
includes for `GL/gl.h` / `OpenGL/gl3.h`.

## Host plugin integration

```cpp
// PluginProcessor
void prepareToPlay (double sr, int) override {
    editor->waterfall.setSampleRate (sr);     // forwards to CudaProcessor::initialize
}

void processBlock (juce::AudioBuffer<float>& buf, juce::MidiBuffer&) override {
    // Sum to mono into a stack scratch (NOT heap) — RT safe
    float mono[2048];
    const int n = juce::jmin (buf.getNumSamples(), 2048);
    const int chans = buf.getNumChannels();
    for (int i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int c = 0; c < chans; ++c) s += buf.getReadPointer (c)[i];
        mono[i] = s / (float) chans;
    }
    editor->waterfall.pushAudioBlock (mono, n);
}
```

UI flips modes via `loadStaticFile(file)` / `switchToLiveStream()`.

---

## Configuration constraints

The compile-time constants in `CudaProcessor.h` are not all free
parameters — some have hard alignment / size requirements baked into
the kernels.

| Constant | Default | Constraint |
|---|---|---|
| `FFT_SIZE` | 2048 | Power of two (cuFFT R2C plan, Hann LUT, ring math) |
| `HOP_SIZE` | 512 | Any positive int ≤ `FFT_SIZE`; 512 = 75 % overlap at FFT 2048 |
| `NUM_OUTPUT_BINS` | 512 | **Power of two, ideally ≥ 128.** See below. |
| `NUM_HISTORY_ROWS` | 256 | Any positive int; affects live VRAM only |
| `PCM_RING_SIZE` | `FFT_SIZE * 4` | Must be ≥ `FFT_SIZE + HOP_SIZE`, power of two preferred |

**Why `NUM_OUTPUT_BINS` must stay power-of-two.** Each mesh row is laid
out contiguously in the VBO as `NUM_OUTPUT_BINS × sizeof(float) =
2048 B`, a multiple of 128 bits and 256 bits. That alignment lets
`spectrumToVboRowLive` issue fully coalesced 128-bit stores straight
into the mapped VBO — one transaction per warp, no scatter. Changing
to e.g. 500 or 768 will still produce correct output but uncoalesces
the VBO writes; expect a 3–5× drop in write throughput and visible
scrolling stutter under load. Going non-aligned would require
`cudaMallocPitch` and a pitched VBO layout that the current interop
path doesn't support.

---

## TODO — validate on first bring-up

- [ ] **VBO size matches.** Vertex count in `buildLiveGeometry` /
  `buildStaticGeometry` must equal `NUM_OUTPUT_BINS * numRows`. The
  CUDA-side `rebindVbo(handle, numRows)` keys off the second arg for
  bounds checks in the bin-write kernel — mismatch will silently
  truncate the spectrum, not crash.

- [ ] **OpenGL context profile is 3.2+ core.** Must be requested
  *before* `attachTo(*this)`. Already done in the component constructor,
  but if you embed this inside another GL-owning parent, make sure the
  outer context isn't compat profile —
  `cudaGraphicsGLRegisterBuffer` returns
  `cudaErrorInvalidGraphicsContext` on a compat context on some driver
  versions.

- [ ] **Surface-normal orientation looks right.** The fragment shader
  derives normals via `cross(dFdx(vWorldPos), dFdy(vWorldPos))`. The
  sign depends on triangle winding × screen-space projection
  orientation; the current orbit camera + CCW mesh should produce
  outward-facing normals, but if the surface comes back uniformly
  dark (diffuse pinned near zero, fog dominating), swap the cross
  argument order to `cross(dFdy(vWorldPos), dFdx(vWorldPos))`. One
  character; flips the normal.

## Future extension

If you want a moving "now" cursor instead of pinning to end-of-file in
static mode, expose `setStaticPlayhead(int rowIdx)` and override
`frameIndex` in the static branch of `renderOpenGL`. The shader already
does the right thing for any value in `[0, numRows-1]`.
