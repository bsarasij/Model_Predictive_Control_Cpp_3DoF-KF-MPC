# Note: Use Visual Studio generator instead of Ninja to avoid compiler environment issues
# In Visual Studio: CMake -> Change CMake Settings -> CMake Settings -> Generator -> Visual Studio 17 2022
cmake_minimum_required(VERSION 3.14)
project(MPC_CPP)

include_directories(${CMAKE_SOURCE_DIR}/include)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Install path" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/include/*.h
    ${PROJECT_SOURCE_DIR}/include/*.hpp
)

# Set CMAKE_PREFIX_PATH so find_package can locate OsqpEigen and OSQP
# Users can override these paths if needed
if(NOT DEFINED OSQP_PREFIX_PATH)
  set(OSQP_PREFIX_PATH "" CACHE PATH "Path to OSQP installation")
endif()
if(NOT DEFINED OSQPEIGEN_PREFIX_PATH)
  set(OSQPEIGEN_PREFIX_PATH "" CACHE PATH "Path to OsqpEigen installation")
endif()

# Set Boost paths - users can override these
if(NOT DEFINED BOOST_ROOT)
  set(BOOST_ROOT "" CACHE PATH "Path to Boost installation")
endif()

# Find Eigen3
find_package(Eigen3 3.4.0 REQUIRED)
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Please install Eigen3 3.4.0 or later.")
endif()

# Find OsqpEigen (which will also find OSQP as a dependency)
find_package(OsqpEigen REQUIRED)
if(NOT OsqpEigen_FOUND)
    message(FATAL_ERROR "OsqpEigen not found. Please install OsqpEigen and OSQP.")
endif()

# Find Boost with iostreams component (required for gnuplot-iostream.h)
find_package(Boost 1.88.0 REQUIRED COMPONENTS iostreams)
if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost 1.88.0 or later with iostreams component not found.")
endif()

# Add sources to the executable. Headers are found via target_include_directories.
add_executable(${PROJECT_NAME} ${SOURCES})

if (MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /bigobj /FS)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    Eigen3::Eigen
    OsqpEigen::OsqpEigen 
    Boost::iostreams
)

target_compile_definitions(${PROJECT_NAME} PRIVATE 
    OSQP_EIGEN_OSQP_IS_V1
    OSQP_EIGEN_OSQP_IS_V1_FINAL
)

message("Project: ${PROJECT_NAME}")
message("C++ Standard: ${CMAKE_CXX_STANDARD}")
message("Build Type: ${CMAKE_BUILD_TYPE}")

# Install the executable
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

# Install dependencies (platform-specific)
if(WIN32)
    # Windows DLL installation - users need to copy these manually or use package managers
    message(STATUS "On Windows, you may need to copy DLLs manually or use vcpkg")
else()
    # Linux/macOS typically handle dependencies through package managers
    message(STATUS "On Linux/macOS, dependencies are typically handled by package managers")
endif()
