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

# ========== Find Eigen (required for MLP math) ========== #
find_package(Eigen3 REQUIRED)

# ========== Core library ========== #
file(GLOB CORE_SOURCES
    ${SRC_DIR}/*.cpp
)

# ========== Add Internal Core Library ========== #
add_library(mlp_core ${CORE_SOURCES})

# ========== Link Libraries to the Internal Core Lobrary ========== #
target_link_libraries(mlp_core PUBLIC Eigen3::Eigen)

# Also expose headers
target_include_directories(mlp_core PUBLIC ${INCLUDE_DIR})

# ========== Main executable (apps/main.cpp) ========== #
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)   # Set executables to live in bin/
add_executable(main ${APPS_DIR}/main.cpp)
target_link_libraries(main PRIVATE mlp_core)

add_executable(trainMLP ${APPS_DIR}/trainMLP.cpp)
target_link_libraries(trainMLP PRIVATE mlp_core)
add_executable(testMLP ${APPS_DIR}/testMLP.cpp)
target_link_libraries(testMLP PRIVATE mlp_core)


# ========== Add when finished with test and train ========== #
# add_executable(train ${APPS_DIR}/train_main.cpp)
# target_link_libraries(train PRIVATE mlp_core)
#
# add_executable(test ${APPS_DIR}/test_main.cpp)
# target_link_libraries(test PRIVATE mlp_core)

