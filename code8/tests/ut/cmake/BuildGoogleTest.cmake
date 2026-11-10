include_guard(GLOBAL)

# nlohmann/json is only needed by the host tiling test harness.
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
          HINTS "$ENV{NLOHMANN_JSON_INCLUDE_DIR}")
if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(FATAL_ERROR
        "nlohmann/json.hpp not found. Set NLOHMANN_JSON_INCLUDE_DIR to the include "
        "directory that contains nlohmann/json.hpp (3.11.3 recommended), e.g.\n"
        "  NLOHMANN_JSON_INCLUDE_DIR=/opt/json/include bash build.sh -u")
endif()
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})

# Keep the UT build simple and deterministic:
#   - Release unless the caller explicitly selected a build type.
#   - Empty DEBUG_POSTFIX so the imported target name (gtest) and the produced
#     archive name (libgtest.a) can never diverge as libgtestd.a.
#   - ABI=0 to match CANN's libstdc++ ABI.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_DEBUG_POSTFIX "" CACHE STRING "No debug postfix for GoogleTest" FORCE)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set(GTEST_SOURCE_DIR "$ENV{GTEST_SOURCE_DIR}" CACHE PATH "GoogleTest source directory")
if(NOT EXISTS "${GTEST_SOURCE_DIR}/CMakeLists.txt" AND EXISTS "/usr/src/googletest/CMakeLists.txt")
    set(GTEST_SOURCE_DIR "/usr/src/googletest" CACHE PATH "GoogleTest source directory" FORCE)
    message(STATUS "Using GoogleTest sources: ${GTEST_SOURCE_DIR}")
endif()

if(EXISTS "${GTEST_SOURCE_DIR}/CMakeLists.txt")
    add_subdirectory(${GTEST_SOURCE_DIR} ${CMAKE_BINARY_DIR}/gtest EXCLUDE_FROM_ALL)
else()
    # System packages are accepted as a fallback, but source compilation from
    # GTEST_SOURCE_DIR is preferred because it can be built with ABI=0.
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(FATAL_ERROR
            "GoogleTest not found. Set GTEST_SOURCE_DIR to a local GoogleTest source "
            "tree (1.14.0 recommended), or install a system libgtest-dev package.")
    endif()
    if(NOT TARGET gtest)
        add_library(gtest INTERFACE)
        target_link_libraries(gtest INTERFACE GTest::gtest)
    endif()
    if(NOT TARGET gtest_main)
        add_library(gtest_main INTERFACE)
        target_link_libraries(gtest_main INTERFACE GTest::gtest_main)
    endif()
endif()

add_library(intf_llt_pub_asan_cxx17 INTERFACE)
target_link_libraries(intf_llt_pub_asan_cxx17 INTERFACE gtest)
