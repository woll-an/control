cmake_minimum_required(VERSION 3.4)
project(control VERSION 0.0.1 LANGUAGES CXX)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

add_library(lqr STATIC lqr.cpp)
target_link_libraries(lqr PRIVATE Eigen3::Eigen3)

add_executable(lqr-tests test.cpp)
target_link_libraries(lqr-tests PUBLIC Eigen3::Eigen3 GTest::GTest GTest::Main lqr)
add_test(Default safety-tests --gtest_output=xml:${TEST_OUTPUT_DIR}/default.xml)