// =============================================================================
//  SpectralWaterfallComponent.cpp
// =============================================================================

#include "SpectralWaterfallComponent.h"

using namespace juce::gl;   // brings glXxx symbols into the file

// =============================================================================
//  GLSL SHADERS (Core Profile 3.30)
// =============================================================================
//
//  VBO layout: a single float per vertex, the height/amplitude in [0, 1]
//  (as written by CUDA). X (frequency bin) and Z (time slice) are derived
//  from gl_VertexID inside the vertex shader so the GPU does not have to
//  touch a "static" position buffer that would otherwise dominate bandwidth.
//
//  Indexed rendering with GL_TRIANGLES walks the (numFreqBins × numRows)
//  grid; the EBO is built once at geometry-creation time.
//
//  The uCurrentFrameIndex uniform implements the zero-copy scrolling
//  effect: each frame the CUDA kernel writes the *new* row into slot
//  (currentFrameIndex % numRows) and we shift the displayed Z so the
//  newest data slides to the front of the camera.
// =============================================================================

static const char* kVertexShaderSrc = R"GLSL(
#version 330 core

// Single float per vertex — the height/amplitude in [0, 1].
layout(location = 0) in float aHeight;

uniform mat4  uMVP;
uniform int   uCurrentFrameIndex;   // most recently written row slot
uniform int   uNumFreqBins;         // X resolution
uniform int   uNumRows;             // Z resolution (history depth)
uniform float uHeightScale;         // vertical amplification
uniform float uTime;                // seconds since component start

out float vHeight;                  // 0..1, passed to fragment
out vec3  vWorldPos;                // world-space position; fragment derives
                                    // surface normal from this via dFdx/dFdy

void main()
{
    int idx       = gl_VertexID;
    int freqBin   = idx % uNumFreqBins;
    int rowIdx    = idx / uNumFreqBins;

    // Circular shift so the most recent row sits at the "front" (close to
    // the camera). (uCurrentFrameIndex - rowIdx) mod uNumRows gives age in
    // rows; mapping age -> -Z brings newest data toward +Z (camera side).
    int age = (uCurrentFrameIndex - rowIdx + uNumRows) % uNumRows;

    // Normalised coordinates. xN in [-1, 1] from low->high frequency.
    // zN in [-1, 1] where +1 = newest (camera side), -1 = oldest (far).
    float xN = (float(freqBin) / float(uNumFreqBins - 1)) * 2.0 - 1.0;
    float zN = 1.0 - (float(age) / float(uNumRows - 1)) * 2.0;

    // World-space placement. The spectrogram lives on a 4 x 4 footprint.
    float x = xN * 2.0;
    float z = zN * 2.0;
    float y = aHeight * uHeightScale;

    vec3 worldPos = vec3(x, y, z);

    // Per-vertex normals deliberately omitted — the fragment shader derives
    // a true per-pixel surface normal from screen-space derivatives of
    // vWorldPos. This keeps the VBO at one float per vertex while giving us
    // proper, continuously varying topology shading instead of a constant
    // fake direction. The RTX 4070 has plenty of headroom for dFdx/dFdy.
    vWorldPos = worldPos;
    vHeight   = aHeight;

    gl_Position = uMVP * vec4(worldPos, 1.0);
}
)GLSL";

static const char* kFragmentShaderSrc = R"GLSL(
#version 330 core

in float vHeight;
in vec3  vWorldPos;

uniform float uTime;

out vec4 FragColor;

// Cyberpunk gradient: deep space navy → cyan → hot pink.
vec3 spectrumColor (float t)
{
    vec3 navy = vec3(0.02, 0.04, 0.18);
    vec3 cyan = vec3(0.00, 0.95, 1.00);
    vec3 pink = vec3(1.00, 0.15, 0.70);

    t = clamp(t, 0.0, 1.0);
    if (t < 0.5)
        return mix(navy, cyan, smoothstep(0.0, 0.5, t));
    else
        return mix(cyan, pink, smoothstep(0.5, 1.0, t));
}

void main()
{
    // Compute real surface normals dynamically using screen-space derivatives.
    // dFdx / dFdy give us the per-pixel rate of change of vWorldPos across
    // adjacent fragments in the 2x2 quad the GPU rasterises in lockstep;
    // their cross product is the true geometric normal of the triangle at
    // this pixel, no per-vertex normal storage required.
    vec3 normal = normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)));

    vec3  baseColor = spectrumColor(vHeight);

    // Subtle neon glow that intensifies at peaks.
    float glow = pow(vHeight, 3.0);
    baseColor += vec3(glow * 0.30, glow * 0.10, glow * 0.35);

    // Dynamic directional shading using the screen-space normal.
    vec3  lightDir = normalize(vec3(0.4, 1.0, 0.6));
    float diffuse  = clamp(dot(normal, lightDir), 0.0, 1.0);
    baseColor *= 0.45 + 0.55 * diffuse;

    // Distance fog toward navy for atmospheric depth.
    float fogAmount = clamp((vWorldPos.z + 2.0) / 4.0, 0.0, 1.0);
    baseColor = mix(vec3(0.02, 0.04, 0.18), baseColor, fogAmount * 0.85 + 0.15);

    FragColor = vec4(baseColor, 1.0);
}
)GLSL";

// =============================================================================
//  Construction / destruction
// =============================================================================
SpectralWaterfallComponent::SpectralWaterfallComponent()
{
    // Attach the GL renderer.
    openGLContext.setRenderer (this);
    openGLContext.setContinuousRepainting (true);
    openGLContext.setComponentPaintingEnabled (false); // we do all the rendering

    // Request a modern Core Profile context. JUCE will pick the highest
    // available version supporting GLSL 330.
    openGLContext.setOpenGLVersionRequired (juce::OpenGLContext::OpenGLVersion::openGL3_2);
    openGLContext.setMultisamplingEnabled (true);
    juce::OpenGLPixelFormat pf;
    pf.multisamplingLevel = 4;
    openGLContext.setPixelFormat (pf);

    openGLContext.attachTo (*this);

    audioFormatManager.registerBasicFormats(); // WAV, AIFF, FLAC, MP3 (if built)
    setOpaque (true);
    setSize (960, 540);
}

SpectralWaterfallComponent::~SpectralWaterfallComponent()
{
    // Detach first: this triggers openGLContextClosing() on the GL thread,
    // where we tear down CUDA and GL resources safely.
    openGLContext.detach();
}

// =============================================================================
//  Host plugin integration
// =============================================================================
void SpectralWaterfallComponent::setSampleRate (double newSampleRate)
{
    sampleRate.store (newSampleRate, std::memory_order_release);
}

int SpectralWaterfallComponent::pushAudioBlock (const float* monoSamples, int numSamples) noexcept
{
    // Hot path. Delegates straight into the lock-free FIFO inside CudaProcessor.
    return cuda.pushAudioBlock (monoSamples, numSamples);
}

bool SpectralWaterfallComponent::loadStaticFile (const juce::File& file)
{
    // UI thread. Decode the file synchronously into a mono buffer, then ask
    // the GL thread to perform the static FFT on its next frame.
    std::unique_ptr<juce::AudioFormatReader> reader (audioFormatManager.createReaderFor (file));
    if (reader == nullptr)
    {
        DBG ("Failed to open audio file: " << file.getFullPathName());
        return false;
    }

    const int  numChans   = static_cast<int> (reader->numChannels);
    const auto numSamples = static_cast<int>  (reader->lengthInSamples);
    if (numSamples <= CudaProcessor::FFT_SIZE)
    {
        DBG ("File too short for FFT.");
        return false;
    }

    // Decode into a 2-channel temp, then sum to mono.
    juce::AudioBuffer<float> tmp (numChans, numSamples);
    reader->read (&tmp, 0, numSamples, 0, true, numChans > 1);

    juce::AudioBuffer<float> mono (1, numSamples);
    mono.clear();
    auto* m = mono.getWritePointer (0);
    for (int ch = 0; ch < numChans; ++ch)
    {
        const auto* src = tmp.getReadPointer (ch);
        for (int i = 0; i < numSamples; ++i)
            m[i] += src[i];
    }
    if (numChans > 1)
    {
        const float scale = 1.0f / static_cast<float> (numChans);
        for (int i = 0; i < numSamples; ++i)
            m[i] *= scale;
    }

    {
        std::lock_guard<std::mutex> g (staticDataMutex);
        staticPcmMono   = std::move (mono);
        staticSampleRate = reader->sampleRate;
    }

    mode.store (Mode::StaticFile, std::memory_order_release);
    pendingStaticUpload.store (true, std::memory_order_release);
    return true;
}

void SpectralWaterfallComponent::switchToLiveStream()
{
    mode.store (Mode::LiveStream, std::memory_order_release);
    pendingLiveRebuild.store (true, std::memory_order_release);
}

// =============================================================================
//  juce::Component
// =============================================================================
void SpectralWaterfallComponent::resized()
{
    // Viewport reset is handled in renderOpenGL via getRenderingScale().
}

// =============================================================================
//  GL context creation
// =============================================================================
void SpectralWaterfallComponent::newOpenGLContextCreated()
{
    // Compile / link the shaders.
    if (! buildShaders())
    {
        jassertfalse;
        return;
    }

    // Build the live-mode geometry by default.
    if (! buildLiveGeometry())
    {
        jassertfalse;
        return;
    }

    // Initialise CUDA against the live VBO.
    if (! cuda.initialize (vboHeights, sampleRate.load()))
    {
        jassertfalse;
        return;
    }

    glEnable (GL_DEPTH_TEST);
    glDepthFunc (GL_LESS);
    glEnable (GL_CULL_FACE);
    glCullFace (GL_BACK);

    // Modest line/point smoothing for the neon look.
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    startTime = juce::Time::getCurrentTime();
}

// =============================================================================
//  Shaders
// =============================================================================
bool SpectralWaterfallComponent::buildShaders()
{
    auto newShader = std::make_unique<juce::OpenGLShaderProgram> (openGLContext);

    if (! newShader->addVertexShader   (kVertexShaderSrc) ||
        ! newShader->addFragmentShader (kFragmentShaderSrc) ||
        ! newShader->link())
    {
        DBG ("Shader compile/link failed: " << newShader->getLastError());
        return false;
    }

    shader = std::move (newShader);
    shader->use();

    uMVP                = std::make_unique<juce::OpenGLShaderProgram::Uniform> (*shader, "uMVP");
    uCurrentFrameIndex  = std::make_unique<juce::OpenGLShaderProgram::Uniform> (*shader, "uCurrentFrameIndex");
    uNumFreqBins        = std::make_unique<juce::OpenGLShaderProgram::Uniform> (*shader, "uNumFreqBins");
    uNumRows            = std::make_unique<juce::OpenGLShaderProgram::Uniform> (*shader, "uNumRows");
    uHeightScale        = std::make_unique<juce::OpenGLShaderProgram::Uniform> (*shader, "uHeightScale");
    uTime               = std::make_unique<juce::OpenGLShaderProgram::Uniform> (*shader, "uTime");

    return true;
}

// =============================================================================
//  Geometry builders
// =============================================================================
//  We use a height-only VBO (one float per vertex). The vertex shader
//  reconstructs (x, y, z) from gl_VertexID + the height.
//
//  Indices form a triangle mesh over (numFreqBins × numRows). Indices are
//  stored as uint32 to safely cover the static-file mode where numHops can
//  easily exceed 65 k for long recordings.
// =============================================================================
namespace
{
    void buildGridIndices (std::vector<uint32_t>& indices, int numFreqBins, int numRows)
    {
        indices.clear();
        indices.reserve (static_cast<std::size_t> ((numFreqBins - 1) * (numRows - 1) * 6));
        for (int r = 0; r < numRows - 1; ++r)
        {
            for (int x = 0; x < numFreqBins - 1; ++x)
            {
                uint32_t i00 = static_cast<uint32_t> (r       * numFreqBins + x);
                uint32_t i10 = static_cast<uint32_t> (r       * numFreqBins + x + 1);
                uint32_t i01 = static_cast<uint32_t> ((r + 1) * numFreqBins + x);
                uint32_t i11 = static_cast<uint32_t> ((r + 1) * numFreqBins + x + 1);
                // Two CCW triangles.
                indices.push_back (i00); indices.push_back (i01); indices.push_back (i10);
                indices.push_back (i10); indices.push_back (i01); indices.push_back (i11);
            }
        }
    }
} // namespace

bool SpectralWaterfallComponent::buildLiveGeometry()
{
    releaseGeometry();

    const int rows = CudaProcessor::NUM_HISTORY_ROWS;
    const int cols = CudaProcessor::NUM_OUTPUT_BINS;
    const std::size_t vertCount = static_cast<std::size_t> (rows) * cols;

    glGenVertexArrays (1, &vao);
    glBindVertexArray (vao);

    glGenBuffers (1, &vboHeights);
    glBindBuffer (GL_ARRAY_BUFFER, vboHeights);
    glBufferData (GL_ARRAY_BUFFER,
                  static_cast<GLsizeiptr> (vertCount * sizeof (float)),
                  nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer (0, 1, GL_FLOAT, GL_FALSE, sizeof (float), nullptr);
    glEnableVertexAttribArray (0);

    std::vector<uint32_t> indices;
    buildGridIndices (indices, cols, rows);
    indexCount = static_cast<int> (indices.size());

    glGenBuffers (1, &ebo);
    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData (GL_ELEMENT_ARRAY_BUFFER,
                  static_cast<GLsizeiptr> (indices.size() * sizeof (uint32_t)),
                  indices.data(), GL_STATIC_DRAW);

    glBindVertexArray (0);
    geometryNumRows = rows;
    return true;
}

bool SpectralWaterfallComponent::buildStaticGeometry (int numRows)
{
    releaseGeometry();

    const int cols = CudaProcessor::NUM_OUTPUT_BINS;
    const std::size_t vertCount = static_cast<std::size_t> (numRows) * cols;

    glGenVertexArrays (1, &vao);
    glBindVertexArray (vao);

    glGenBuffers (1, &vboHeights);
    glBindBuffer (GL_ARRAY_BUFFER, vboHeights);
    glBufferData (GL_ARRAY_BUFFER,
                  static_cast<GLsizeiptr> (vertCount * sizeof (float)),
                  nullptr, GL_STATIC_DRAW);
    glVertexAttribPointer (0, 1, GL_FLOAT, GL_FALSE, sizeof (float), nullptr);
    glEnableVertexAttribArray (0);

    std::vector<uint32_t> indices;
    buildGridIndices (indices, cols, numRows);
    indexCount = static_cast<int> (indices.size());

    glGenBuffers (1, &ebo);
    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData (GL_ELEMENT_ARRAY_BUFFER,
                  static_cast<GLsizeiptr> (indices.size() * sizeof (uint32_t)),
                  indices.data(), GL_STATIC_DRAW);

    glBindVertexArray (0);
    geometryNumRows = numRows;

    // Re-register the new VBO with CUDA.
    return cuda.rebindVbo (vboHeights, numRows);
}

void SpectralWaterfallComponent::releaseGeometry()
{
    if (vao)        { glDeleteVertexArrays (1, &vao);        vao        = 0; }
    if (vboHeights) { glDeleteBuffers      (1, &vboHeights); vboHeights = 0; }
    if (ebo)        { glDeleteBuffers      (1, &ebo);        ebo        = 0; }
    indexCount = 0;
    geometryNumRows = 0;
}

// =============================================================================
//  Camera
// =============================================================================
void SpectralWaterfallComponent::setCameraUniforms()
{
    auto bounds   = getLocalBounds();
    auto desktopScale = (float) openGLContext.getRenderingScale();
    const int w = juce::jmax (1, (int) (bounds.getWidth()  * desktopScale));
    const int h = juce::jmax (1, (int) (bounds.getHeight() * desktopScale));
    glViewport (0, 0, w, h);

    const float aspect = (float) w / (float) h;

    // Slow orbit so the user can see the 3D structure.
    const float t = (float) (juce::Time::getCurrentTime() - startTime).inSeconds();
    cameraOrbit = 0.15f * t;

    const float camRadius = 6.0f;
    const float camHeight = 3.2f;
    const float camX      = std::sin (cameraOrbit) * camRadius;
    const float camZ      = std::cos (cameraOrbit) * camRadius;

    juce::Vector3D<float> eye    (camX, camHeight, camZ);
    juce::Vector3D<float> target (0.0f, 0.4f, 0.0f);
    juce::Vector3D<float> up     (0.0f, 1.0f, 0.0f);

    // Build view (lookAt) matrix.
    auto lookAt = [] (juce::Vector3D<float> e,
                      juce::Vector3D<float> c,
                      juce::Vector3D<float> u) -> juce::Matrix3D<float>
    {
        auto f = (c - e); f /= f.length();
        auto s = f ^ u;   s /= s.length();
        auto v = s ^ f;

        juce::Matrix3D<float> m;
        // Column-major mat4
        float* d = m.mat;
        d[ 0] =  s.x;  d[ 1] =  v.x;  d[ 2] = -f.x;  d[ 3] = 0.0f;
        d[ 4] =  s.y;  d[ 5] =  v.y;  d[ 6] = -f.y;  d[ 7] = 0.0f;
        d[ 8] =  s.z;  d[ 9] =  v.z;  d[10] = -f.z;  d[11] = 0.0f;
        d[12] = -(s.x * e.x + s.y * e.y + s.z * e.z);
        d[13] = -(v.x * e.x + v.y * e.y + v.z * e.z);
        d[14] =  (f.x * e.x + f.y * e.y + f.z * e.z);
        d[15] = 1.0f;
        return m;
    };

    auto view = lookAt (eye, target, up);

    auto proj = juce::Matrix3D<float>::fromFrustum (-aspect * 0.05f, aspect * 0.05f,
                                                    -0.05f, 0.05f,
                                                     0.1f, 100.0f);

    // GLSL is column-vector: gl_Position = uMVP * pos. So uMVP must be
    // projection * view (applies view first, then projection).
    auto mvp = proj * view;

    if (uMVP != nullptr)               uMVP->setMatrix4 (mvp.mat, 1, false);
    if (uNumFreqBins != nullptr)       uNumFreqBins->set ((GLint) CudaProcessor::NUM_OUTPUT_BINS);
    if (uNumRows != nullptr)           uNumRows->set ((GLint) geometryNumRows);
    if (uHeightScale != nullptr)       uHeightScale->set (1.4f);
    if (uTime != nullptr)              uTime->set (t);
}

// =============================================================================
//  renderOpenGL — the per-frame driver
// =============================================================================
void SpectralWaterfallComponent::renderOpenGL()
{
    if (shader == nullptr) return;

    // ---- Handle pending mode transitions (GL thread, context current) ------
    if (pendingLiveRebuild.exchange (false, std::memory_order_acq_rel))
    {
        buildLiveGeometry();
        // The cuda processor's currently registered VBO is whatever the last
        // mode set up. Rebind to the new live VBO.
        cuda.rebindVbo (vboHeights, CudaProcessor::NUM_HISTORY_ROWS);
    }

    if (pendingStaticUpload.exchange (false, std::memory_order_acq_rel))
    {
        juce::AudioBuffer<float> localCopy;
        double sr = 44100.0;
        {
            std::lock_guard<std::mutex> g (staticDataMutex);
            localCopy = std::move (staticPcmMono);  // move: no deep copy under the lock
            sr        = staticSampleRate;
        }

        if (localCopy.getNumSamples() > CudaProcessor::FFT_SIZE)
        {
            const int totalSamples = localCopy.getNumSamples();
            const int numHops      = (totalSamples - CudaProcessor::FFT_SIZE)
                                   / CudaProcessor::HOP_SIZE + 1;

            buildStaticGeometry (numHops);  // resizes VBO + re-registers with CUDA

            int outRows = 0;
            if (! cuda.processStaticFile (localCopy.getReadPointer (0),
                                          static_cast<std::size_t> (totalSamples),
                                          sr,
                                          outRows))
            {
                DBG ("processStaticFile failed — VBO may be empty.");
            }
            // Static mode renders with frame index = 0 (no scrolling).
        }
    }

    // ---- Clear --------------------------------------------------------------
    glClearColor (0.01f, 0.015f, 0.04f, 1.0f);    // near-black navy
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    shader->use();
    setCameraUniforms();

    // ---- Drive CUDA --------------------------------------------------------
    //
    //  LiveStream: invoke the GPU pipeline, which returns the row slot that
    //  was just written. The vertex shader treats that slot as "age 0", so
    //  the freshest hop sits at z = +1 (closest to the camera).
    //
    //  StaticFile: the VBO is already fully populated. We pin the frame
    //  index to (numRows - 1) so the END of the file (the most recent hop
    //  in playback time) maps to age 0 and lands at the camera, giving the
    //  natural "time flows toward you" reading along -Z.
    int frameIndex = 0;
    if (mode.load (std::memory_order_acquire) == Mode::LiveStream)
        frameIndex = cuda.renderLiveFrame();
    else if (geometryNumRows > 0)
        frameIndex = geometryNumRows - 1;

    if (uCurrentFrameIndex != nullptr)
        uCurrentFrameIndex->set ((GLint) frameIndex);

    // ---- Draw --------------------------------------------------------------
    if (vao != 0 && indexCount > 0)
    {
        glBindVertexArray (vao);
        glDrawElements (GL_TRIANGLES, (GLsizei) indexCount, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray (0);
    }
}

// =============================================================================
//  GL context tear-down
// =============================================================================
void SpectralWaterfallComponent::openGLContextClosing()
{
    // Order matters: tear down CUDA first (it still holds a reference to the
    // GL VBO via cudaGraphicsResource).
    cuda.shutdown();

    releaseGeometry();
    shader.reset();

    uMVP.reset();
    uCurrentFrameIndex.reset();
    uNumFreqBins.reset();
    uNumRows.reset();
    uHeightScale.reset();
    uTime.reset();
}
