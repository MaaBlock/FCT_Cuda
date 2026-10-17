#include "ImageKernels.h"
#include <device_launch_parameters.h>
#include <cuda_surface_types.h>
#include <cmath>

namespace FCT::Cuda::Kernels {

    __global__ void SurfaceGradientKernel(
        cudaSurfaceObject_t surface,
        uint32_t width,
        uint32_t height,
        float time) {
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        float u = static_cast<float>(x) / static_cast<float>(width);
        float v = static_cast<float>(y) / static_cast<float>(height);

        float r = 0.5f + 0.5f * sinf(u * 10.0f + time);
        float g = 0.5f + 0.5f * sinf(v * 10.0f + time * 1.3f);
        float b = 0.5f + 0.5f * cosf((u + v) * 5.0f + time);

        uchar4 pixel;
        pixel.x = static_cast<unsigned char>(__saturatef(r) * 255.0f);
        pixel.y = static_cast<unsigned char>(__saturatef(g) * 255.0f);
        pixel.z = static_cast<unsigned char>(__saturatef(b) * 255.0f);
        pixel.w = 255;

        surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
    }

    __global__ void BufferGradientKernel(
        unsigned char* devicePtr,
        size_t pitch,
        uint32_t width,
        uint32_t height,
        float time) {
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        float u = static_cast<float>(x) / static_cast<float>(width);
        float v = static_cast<float>(y) / static_cast<float>(height);

        float r = 0.5f + 0.5f * sinf(u * 8.0f + time * 2.0f);
        float g = 0.5f + 0.5f * sinf(v * 8.0f + time * 1.5f);
        float b = 0.5f + 0.5f * cosf((u + v) * 4.0f + time);

        uchar4* row = reinterpret_cast<uchar4*>(devicePtr + y * pitch);
        uchar4 pixel;
        pixel.x = static_cast<unsigned char>(__saturatef(r) * 255.0f);
        pixel.y = static_cast<unsigned char>(__saturatef(g) * 255.0f);
        pixel.z = static_cast<unsigned char>(__saturatef(b) * 255.0f);
        pixel.w = 255;

        row[x] = pixel;
    }

    __global__ void BufferColorFillKernel(
        unsigned char* devicePtr,
        size_t pitch,
        uint32_t width,
        uint32_t height,
        float r,
        float g,
        float b,
        float a) {
        uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
        uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        uchar4* row = reinterpret_cast<uchar4*>(devicePtr + y * pitch);
        uchar4 pixel;
        pixel.x = static_cast<unsigned char>(__saturatef(r) * 255.0f);
        pixel.y = static_cast<unsigned char>(__saturatef(g) * 255.0f);
        pixel.z = static_cast<unsigned char>(__saturatef(b) * 255.0f);
        pixel.w = static_cast<unsigned char>(__saturatef(a) * 255.0f);

        row[x] = pixel;
    }

    void LaunchSurfaceGradientKernel(
        cudaSurfaceObject_t surface,
        uint32_t width,
        uint32_t height,
        float time,
        cudaStream_t stream) {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        SurfaceGradientKernel<<<grid, block, 0, stream>>>(surface, width, height, time);
    }

    void LaunchBufferGradientKernel(
        void* devicePtr,
        size_t pitch,
        uint32_t width,
        uint32_t height,
        float time,
        cudaStream_t stream) {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        BufferGradientKernel<<<grid, block, 0, stream>>>(
            static_cast<unsigned char*>(devicePtr),
            pitch,
            width,
            height,
            time);
    }

    void LaunchBufferColorFillKernel(
        void* devicePtr,
        size_t pitch,
        uint32_t width,
        uint32_t height,
        float r,
        float g,
        float b,
        float a,
        cudaStream_t stream) {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        BufferColorFillKernel<<<grid, block, 0, stream>>>(
            static_cast<unsigned char*>(devicePtr),
            pitch,
            width,
            height,
            r,
            g,
            b,
            a);
    }

} // namespace FCT::Cuda::Kernels
