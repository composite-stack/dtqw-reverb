cmake_minimum_required(VERSION 3.22)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" AXIOM_RELEASE LIMIT_COUNT 1)
string(STRIP "${AXIOM_RELEASE}" AXIOM_RELEASE)
if(NOT AXIOM_RELEASE MATCHES "^[0-9]+$")
  message(FATAL_ERROR "VERSION must contain the numeric source release")
endif()
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/RELEASE_DATE" AXIOM_RELEASE_DATE LIMIT_COUNT 1)
string(STRIP "${AXIOM_RELEASE_DATE}" AXIOM_RELEASE_DATE)

# Universal Binary: Apple Silicon (arm64) + Intel (x86_64), runs on macOS 10.13+
set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "" FORCE)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "" FORCE)

project(AxiomDTQW VERSION 1.0.${AXIOM_RELEASE} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

configure_file(Source/ReleaseVersion.h.in generated/ReleaseVersion.h @ONLY)

# JUCE: build.sh passes a cached clone via FETCHCONTENT_SOURCE_DIR_JUCE; otherwise it is fetched.
include(FetchContent)
FetchContent_Declare(JUCE GIT_REPOSITORY https://github.com/juce-framework/JUCE.git GIT_TAG 8.0.12 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(JUCE)

juce_add_plugin(AxiomDTQW
  PRODUCT_NAME "AXIOM DTQW"
  COMPANY_NAME "Axiom Audio"
  BUNDLE_ID "audio.axiom.dtqw"
  PLUGIN_MANUFACTURER_CODE Axau
  PLUGIN_CODE AxQw
  FORMATS VST3 AU Standalone
  IS_SYNTH FALSE
  NEEDS_MIDI_INPUT TRUE
  NEEDS_MIDI_OUTPUT FALSE
  IS_MIDI_EFFECT FALSE
  EDITOR_WANTS_KEYBOARD_FOCUS FALSE
  COPY_PLUGIN_AFTER_BUILD FALSE
  VST3_CATEGORIES Fx Reverb
  AU_MAIN_TYPE kAudioUnitType_MusicEffect
  MICROPHONE_PERMISSION_ENABLED TRUE
  MICROPHONE_PERMISSION_TEXT "AXIOM DTQW verarbeitet das Eingangssignal in der Standalone-Version.")

juce_generate_juce_header(AxiomDTQW)
target_sources(AxiomDTQW PRIVATE Source/PluginProcessor.cpp Source/PluginEditor.cpp)
target_include_directories(AxiomDTQW PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated" "${CMAKE_CURRENT_SOURCE_DIR}/Source/Dsp")
target_compile_definitions(AxiomDTQW PUBLIC
  JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0 JUCE_DISPLAY_SPLASH_SCREEN=0)
target_link_libraries(AxiomDTQW
  PRIVATE juce::juce_audio_utils juce::juce_dsp
  PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_lto_flags juce::juce_recommended_warning_flags)
