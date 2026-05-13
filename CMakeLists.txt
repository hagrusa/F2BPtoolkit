cmake_minimum_required(VERSION 3.15)
project(f2bptoolkit CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── pybind11 ──────────────────────────────────────────────────────────────────
find_package(pybind11 CONFIG REQUIRED)

# ── Armadillo ─────────────────────────────────────────────────────────────────
find_package(Armadillo REQUIRED)

# ── BLAS/LAPACK (Armadillo needs one) ────────────────────────────────────────
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        message(STATUS "Using Apple Accelerate framework")
        set(BLAS_LIBS ${ACCELERATE_FRAMEWORK})
    endif()
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(BLAS_LIBS ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

# ── C++ extension module ──────────────────────────────────────────────────────
pybind11_add_module(_core
    src/gubas_core.cpp
    src/bindings.cpp
)

target_include_directories(_core PRIVATE
    ${ARMADILLO_INCLUDE_DIRS}
    src/
)

target_link_libraries(_core PRIVATE
    ${ARMADILLO_LIBRARIES}
    ${BLAS_LIBS}
)

target_compile_options(_core PRIVATE
    $<$<CONFIG:Release>:-O3 -DARMA_NO_DEBUG>
    $<$<CONFIG:Debug>:-O0 -g>
)

# Suppress Armadillo print-on-error for library use
target_compile_definitions(_core PRIVATE
    ARMA_DONT_PRINT_ERRORS
)

# ── Install into the Python package directory ─────────────────────────────────
install(TARGETS _core
    LIBRARY DESTINATION f2bptoolkit
    COMPONENT PlatLib
)
