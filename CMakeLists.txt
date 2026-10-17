cmake_minimum_required(VERSION 3.22.1)

include(platform/platform.cmake)
include(../../Config/set_utf8_compiler_encoding.cmake OPTIONAL)

if(MSVC AND MSVC_VERSION GREATER_EQUAL 1940 AND DEFINED ENV{CUDA_PATH_V13_0})
    file(TO_CMAKE_PATH "$ENV{CUDA_PATH_V13_0}" _cuda_v13_root)
    if(EXISTS "${_cuda_v13_root}/bin/nvcc.exe")
        set(CUDAToolkit_ROOT "${_cuda_v13_root}" CACHE PATH "CUDA Toolkit root directory" FORCE)
        set(CMAKE_CUDA_COMPILER "${_cuda_v13_root}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

file(GLOB_RECURSE FCT_CUDA_SOURCES
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "./src/*.cpp"
    "./src/*.cu"
)

file(GLOB_RECURSE FCT_CUDA_HEADERS
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "./src/*.h"
    "./src/*.hpp"
)

add_library(FCT_Cuda STATIC
    ${FCT_CUDA_SOURCES}
    ${FCT_CUDA_HEADERS}
)

if(COMMAND set_utf8_for_target)
    set_utf8_for_target(FCT_Cuda)
endif()

set_target_properties(FCT_Cuda PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/FCT_Cuda)

foreach(header_file ${FCT_CUDA_HEADERS})
    file(RELATIVE_PATH relative_path ${CMAKE_CURRENT_SOURCE_DIR}/src ${header_file})
    get_filename_component(header_dir ${relative_path} DIRECTORY)

    if(header_dir)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/FCT_Cuda/${header_dir})
    endif()

    configure_file(${header_file} ${CMAKE_CURRENT_SOURCE_DIR}/include/FCT_Cuda/${relative_path} COPYONLY)
endforeach()

target_include_directories(FCT_Cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../FCT/include/FCT>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../FCT/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(FCT_Cuda PUBLIC
    FCT
    CUDA::cudart
    CUDA::cuda_driver
)

function(target_link_fct_cuda_libraries target_name)
    target_link_libraries(${target_name} FCT_Cuda)
endfunction()

