cmake_minimum_required(VERSION 3.16)
project(DTQWReverb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DTQW_ENABLE_SANITIZERS "Enable Address and UndefinedBehavior sanitizers for test targets" ON)
option(DTQW_ENABLE_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

# Compiler-Flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wno-deprecated-declarations)
    if(DTQW_ENABLE_WARNINGS_AS_ERRORS)
        add_compile_options(-Werror)
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
        add_compile_options(-msse4.1)
    endif()

    if(DTQW_ENABLE_SANITIZERS)
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

enable_testing()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 1. Worst-Case Stress Tests (Mandatory CI Gate)
add_executable(dtqw_worst_case_stress_tests DTQWWorstCaseStressTests.cpp)
add_test(NAME WorstCaseStressTests COMMAND dtqw_worst_case_stress_tests)
set_tests_properties(WorstCaseStressTests PROPERTIES LABELS "mandatory;ci;stress;allpass;crossfade;sanitizers")

# 2. State Crossfade Tests
add_executable(dtqw_state_crossfade_tests DTQWStateCrossfadeTests.cpp)
add_test(NAME StateCrossfadeTests COMMAND dtqw_state_crossfade_tests)
set_tests_properties(StateCrossfadeTests PROPERTIES LABELS "mandatory;ci;crossfade;sanitizers")

# 3. Dispersion Tests
add_executable(dtqw_dispersion_tests DTQWDispersionTests.cpp)
add_test(NAME DispersionTests COMMAND dtqw_dispersion_tests)
set_tests_properties(DispersionTests PROPERTIES LABELS "mandatory;ci;allpass;dispersion;sanitizers")

# Custom Target: allpass_crossfade_ci
add_custom_target(allpass_crossfade_ci
    COMMAND ${CMAKE_CTEST_COMMAND} -L "mandatory" --output-on-failure
    DEPENDS dtqw_worst_case_stress_tests dtqw_state_crossfade_tests dtqw_dispersion_tests
)

# 4. Offline WAV render harness
add_executable(dtqw_render_wav dtqw_render_wav.cpp)
target_include_directories(dtqw_render_wav PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dtqw_render_wav PRIVATE -O3 -DNDEBUG -fno-sanitize=all)
    target_link_options(dtqw_render_wav PRIVATE -fno-sanitize=all)
endif()

# WAV file I/O tests (kept independent of sanitizer hooks in DSP stress tests)
add_executable(dtqw_wav_harness_tests DTQWWavHarnessTests.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dtqw_wav_harness_tests PRIVATE -UNDEBUG)
endif()
add_test(NAME WavHarnessTests COMMAND dtqw_wav_harness_tests)
set_tests_properties(WavHarnessTests PROPERTIES LABELS "ci;wav;harness")

# 5. Performance Benchmark Executable (Production -O3, ohne Sanitizer)
add_executable(dtqw_benchmark DTQWPerformanceBenchmark.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dtqw_benchmark PRIVATE -O3 -DNDEBUG -fno-sanitize=all)
    target_link_options(dtqw_benchmark PRIVATE -fno-sanitize=all)
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    add_custom_target(check_performance
        COMMAND dtqw_benchmark --json ${CMAKE_CURRENT_BINARY_DIR}/benchmark_current.json
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_performance_regression.py --current ${CMAKE_CURRENT_BINARY_DIR}/benchmark_current.json --baseline ${CMAKE_CURRENT_SOURCE_DIR}/performance_baseline.json --tolerance 25.0
        DEPENDS dtqw_benchmark
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running reproducible DSP performance benchmark and regression check..."
    )
endif()

# 6. JUCE plug-in standalone parity verification
add_executable(dtqw_juce_parity juce/tests/parity_main.cpp)
target_include_directories(dtqw_juce_parity PRIVATE /juce/Source/Dsp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dtqw_juce_parity PRIVATE -O2 -msse4.1)
endif()
add_test(NAME JuceParity COMMAND dtqw_juce_parity)
set_tests_properties(JuceParity PROPERTIES LABELS "mandatory;ci;juce;parity")
