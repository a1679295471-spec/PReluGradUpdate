include_guard(GLOBAL)

# nlohmann/json: allow -DNLOHMANN_JSON_INCLUDE_DIR=..., $NLOHMANN_JSON_INCLUDE_DIR,
# or a system installation.
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp HINTS "$ENV{NLOHMANN_JSON_INCLUDE_DIR}")
if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(FATAL_ERROR
        "nlohmann/json.hpp not found. Set NLOHMANN_JSON_INCLUDE_DIR to the include "
        "directory that contains nlohmann/json.hpp (3.11.3 recommended), e.g.\n"
        "  NLOHMANN_JSON_INCLUDE_DIR=/opt/json/include bash build.sh -u")
endif()
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})

# GoogleTest is added as a subdirectory below and inherits the directory state
# at that point.  The build type (Debug appends the "d" postfix to the library
# names) and the libstdc++ ABI must therefore be fixed *before* it is added,
# otherwise the UT targets require libgtestd.a while gtest builds libgtest.a.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set(GTEST_SOURCE_DIR "$ENV{GTEST_SOURCE_DIR}" CACHE PATH "GoogleTest source directory")
if(NOT EXISTS "${GTEST_SOURCE_DIR}/CMakeLists.txt" AND EXISTS "/usr/src/googletest/CMakeLists.txt")
    # Distribution source packages drop GoogleTest here (Ubuntu: libgtest-dev,
    # openEuler: googletest-devel).  GoogleTest has to be compiled from source
    # because the UT uses _GLIBCXX_USE_CXX11_ABI=0 to match CANN, while the
    # prebuilt system libgtest.a is ABI=1 and would fail to link.
    set(GTEST_SOURCE_DIR "/usr/src/googletest" CACHE PATH "GoogleTest source directory" FORCE)
    message(STATUS "Using GoogleTest sources: ${GTEST_SOURCE_DIR}")
endif()

if(EXISTS "${GTEST_SOURCE_DIR}/CMakeLists.txt")
    add_subdirectory(${GTEST_SOURCE_DIR} ${CMAKE_BINARY_DIR}/gtest EXCLUDE_FROM_ALL)
else()
    # Fall back to a system GoogleTest installation (e.g. apt install libgtest-dev).
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(FATAL_ERROR "Set GTEST_SOURCE_DIR to local GoogleTest sources (1.14.0 recommended), "
                            "or install system GoogleTest (libgtest-dev).")
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

