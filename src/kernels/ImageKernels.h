#ifndef FCT_CUDA_IMAGE_KERNELS_H
#define FCT_CUDA_IMAGE_KERNELS_H

#include <cuda_runtime.h>
#include <cstdint>

namespace FCT::Cuda::Kernels {

    void LaunchSurfaceGradientKernel(
        cudaSurfaceObject_t surface,
        uint32_t width,
        uint32_t height,
        float time,
        cudaStream_t stream = nullptr);

    void LaunchBufferGradientKernel(
        void* devicePtr,
        size_t pitch,
        uint32_t width,
        uint32_t height,
        float time,
        cudaStream_t stream = nullptr);

    void LaunchBufferColorFillKernel(
        void* devicePtr,
        size_t pitch,
        uint32_t width,
        uint32_t height,
        float r,
        float g,
        float b,
        float a,
        cudaStream_t stream = nullptr);

} // namespace FCT::Cuda::Kernels

#endif // FCT_CUDA_IMAGE_KERNELS_H
