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

set(GTEST_SOURCE_DIR "$ENV{GTEST_SOURCE_DIR}" CACHE PATH "GoogleTest source directory")
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

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_library(intf_llt_pub_asan_cxx17 INTERFACE)
target_link_libraries(intf_llt_pub_asan_cxx17 INTERFACE gtest)

