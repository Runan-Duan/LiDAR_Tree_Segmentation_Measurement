cmake_minimum_required(VERSION 3.5)
project(LiDAR_Tree_Segmentation_Measurement)

# Specify that you're using vcpkg for dependency management
set(CMAKE_TOOLCHAIN_FILE "C:/Users/drn20/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "" FORCE)

# Set CMAKE_PREFIX_PATH for vcpkg's installed Eigen3
list(APPEND CMAKE_PREFIX_PATH "C:/Users/drn20/vcpkg/installed/x64-windows/share/eigen3")

set(PCL_DIR "C:/Users/drn20/vcpkg/installed/x64-windows/share/pcl")

# Add an extension path in CMake
set(EXTERNAL_LIB_PATH "C:/Users/drn20/vcpkg/installed/x64-windows/bin")
include_directories(${EXTERNAL_LIB_PATH})
link_directories(${EXTERNAL_LIB_PATH})

# Find PCL package installed by vcpkg
find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
    message(STATUS "Found ZLIB: ${ZLIB_INCLUDE_DIR} ${ZLIB_LIBRARY}")
else()
    message(FATAL_ERROR "ZLIB not found!")
endif()

find_package(Boost REQUIRED COMPONENTS iostreams)
find_package(PCL REQUIRED)


# Include directories for PCL
include_directories(${PCL_INCLUDE_DIRS})

# Add definitions for PCL
add_definitions(${PCL_DEFINITIONS})

# Set up the executable
add_executable(data_small_output main.cpp)

# Link libraries with PCL
target_link_libraries(data_small_output ${PCL_LIBRARIES})
