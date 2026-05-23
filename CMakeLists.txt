cmake_minimum_required(VERSION 3.22)
project(SpectralWaterfall VERSION 1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# JUCE — point JUCE_PATH at your local JUCE root.
# ---------------------------------------------------------------------------
set(JUCE_PATH "C:/Users/julia/Downloads/juce-8.0.13-windows/JUCE"
    CACHE PATH "Path to the JUCE root directory")

add_subdirectory("${JUCE_PATH}" JUCE EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# CUDA toolkit
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# CudaProcessor — compiled exclusively by nvcc.
# No JUCE headers are needed here; the facade (CudaProcessor.h) is pure C++.
# ---------------------------------------------------------------------------
add_library(CudaProcessor STATIC CudaProcessor.cu)

set(SPECTRAL_CUDA_ARCH "89" CACHE STRING
    "CUDA architecture(s) to compile for (e.g. 89 for Ada, 75 for Turing, 86 for Ampere)")

set_target_properties(CudaProcessor PROPERTIES
    CUDA_ARCHITECTURES         ${SPECTRAL_CUDA_ARCH}
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE  ON      # required when linked into a DLL plugin
)

target_include_directories(CudaProcessor PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(CudaProcessor PRIVATE CUDA::cudart CUDA::cufft)

# ---------------------------------------------------------------------------
# JUCE plugin target — compiled by the host C++ compiler (MSVC on Windows).
# ---------------------------------------------------------------------------
juce_add_plugin(SpectralWaterfall
    PLUGIN_MANUFACTURER_CODE Ju1a
    PLUGIN_CODE              Swfv
    FORMATS                  VST3 Standalone
    PRODUCT_NAME             "Spectral Waterfall"
    COMPANY_NAME             "Julia"
    IS_SYNTH                 FALSE
    NEEDS_MIDI_INPUT         FALSE
    NEEDS_MIDI_OUTPUT        FALSE
    IS_MIDI_EFFECT           FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD  FALSE
)

target_sources(SpectralWaterfall
    PRIVATE
        PluginProcessor.cpp
        PluginEditor.cpp
        SpectralWaterfallComponent.cpp
        AboutComponent.cpp
)

target_compile_definitions(SpectralWaterfall
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(SpectralWaterfall
    PRIVATE
        juce::juce_audio_processors
        juce::juce_audio_devices        # required by Standalone wrapper
        juce::juce_audio_utils          # required by Standalone wrapper
        juce::juce_opengl
        juce::juce_audio_formats
        juce::juce_audio_basics
        juce::juce_gui_basics
        juce::juce_core
        CudaProcessor
        CUDA::cudart
        CUDA::cufft
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
)
