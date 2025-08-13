cmake_minimum_required(VERSION 3.15)
project(mlp_mnist LANGUAGES CXX)

# ========== Compiler settings ========== #
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # keep it clean

# ========== Paths ========== #
set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(APPS_DIR ${PROJECT_SOURCE_DIR}/apps)

include_directories(${INCLUDE_DIR})

# ========== Find Eigen ========== #
find_package(Eigen3 REQUIRED)

# ========== SFML for GUI ========== #
#find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# ========== Core library ========== #
file(GLOB CORE_SOURCES
    ${SRC_DIR}/*.cpp)

# ========== Add Internal Core Library ========== #
add_library(mlp_core ${CORE_SOURCES})
target_include_directories(mlp_core PRIVATE ${SFML_INCLUDE_DIRS})
target_link_libraries(mlp_core PRIVATE sfml-graphics sfml-window sfml-system)

# ========== Link Libraries to the Internal Core Lobrary ========== #
target_link_libraries(mlp_core PUBLIC Eigen3::Eigen)

# Also expose headers
target_include_directories(mlp_core PUBLIC ${INCLUDE_DIR})

# ========== Main executables ========== #
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)   # Set executables to live in bin/

add_executable(main ${APPS_DIR}/main.cpp)
target_link_libraries(main PRIVATE mlp_core)

add_executable(trainMLP ${APPS_DIR}/trainMLP.cpp)
target_link_libraries(trainMLP PRIVATE mlp_core)

add_executable(testMLP ${APPS_DIR}/testMLP.cpp)
target_link_libraries(testMLP PRIVATE mlp_core)

add_executable(guiMLP ${APPS_DIR}/guiMLP.cpp)
target_link_libraries(guiMLP PRIVATE mlp_core)
target_link_libraries(guiMLP PRIVATE sfml-graphics sfml-window sfml-system)

# add_executable(test ${APPS_DIR}/test_main.cpp)
# target_link_libraries(test PRIVATE mlp_core)
