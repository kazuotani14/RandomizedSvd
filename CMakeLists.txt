cmake_minimum_required(VERSION 3.5)
project(RandomizedSvd)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

find_package(OpenCV REQUIRED)
include_directories("eigen")

add_executable(TryRandomizedSvd try_randomized_svd.cpp)

add_executable(BackgroundRemoval background_removal.cpp)
target_link_libraries(BackgroundRemoval ${OpenCV_LIBS})
