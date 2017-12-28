cmake_minimum_required(VERSION 3.5)
project(RandomizedSvd)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

find_package(OpenCV REQUIRED)
include_directories("eigen")

# find_package (Eigen3 3.3 REQUIRED NO_MODULE)
# add_executable (example example.cpp)
# target_link_libraries (example Eigen3::Eigen)

add_executable(TryRandomizedSvd try_randomized_svd.cpp)

add_executable(BackgroundRemoval background_removal.cpp)
target_link_libraries(BackgroundRemoval ${OpenCV_LIBS})
