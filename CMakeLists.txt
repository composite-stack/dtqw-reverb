cmake_minimum_required(VERSION 3.16)
project(DTQWReverb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DTQW_ENABLE_SANITIZERS "Enable Address and UndefinedBehavior sanitizers" ON)
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
