cmake_minimum_required(VERSION 3.22.1)

include(platform/platform.cmake)
include(../Config/set_utf8_compiler_encoding.cmake OPTIONAL)

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

if(FCT_CUDA_BUILD_EXAMPLES)
    set(FCT_CUDA_EXAMPLE_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/example/main.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/example/HomeLbm3DPass.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/example/HomeLbm3DPass.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/example/HomeLbm3DSolver.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/example/HomeLbm3DSolver.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/example/msvc_stl_shims.cpp"
    )

    add_executable(fct_cuda_homelbm_3d_example
        ${FCT_CUDA_EXAMPLE_SOURCES}
    )

    if(COMMAND set_utf8_for_target)
        set_utf8_for_target(fct_cuda_homelbm_3d_example)
    endif()

    if(MSVC)
        target_compile_options(fct_cuda_homelbm_3d_example PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        )
    endif()

    set_target_properties(fct_cuda_homelbm_3d_example PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    target_include_directories(fct_cuda_homelbm_3d_example PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/example
    )

    target_link_libraries(fct_cuda_homelbm_3d_example PRIVATE
        FCT
        FCT_Cuda
    )


endif()
