#ifndef FCT_CUDA_INTEROP_IMAGE_H
#define FCT_CUDA_INTEROP_IMAGE_H

#include "CudaCommon.h"
#include "CudaStream.h"
#include "Context/Image.h"

namespace FCT::Cuda {

    enum class InteropMode {
        DirectSurface,
        DeviceBufferStaging
    };

    class CudaInteropImage {
    public:
        explicit CudaInteropImage(Image* image);
        ~CudaInteropImage();

        CudaInteropImage(const CudaInteropImage&) = delete;
        CudaInteropImage& operator=(const CudaInteropImage&) = delete;

        Status initialize(InteropMode mode = InteropMode::DeviceBufferStaging);
        Status release();

        StatusOr<cudaSurfaceObject_t> acquireSurface(cudaStream_t stream = nullptr);
        Status releaseSurface(cudaStream_t stream = nullptr);

        StatusOr<void*> acquireDeviceBuffer(size_t* outPitch = nullptr);
        Status releaseDeviceBuffer(cudaStream_t stream = nullptr);

        Status copyToFctImage(cudaStream_t stream = nullptr);
        Status copyFromFctImage(const void* srcData, size_t dataSize, cudaStream_t stream = nullptr);

        [[nodiscard]] Image* getFctImage() const noexcept {
            return m_image;
        }

        [[nodiscard]] uint32_t width() const noexcept {
            return m_width;
        }

        [[nodiscard]] uint32_t height() const noexcept {
            return m_height;
        }

        [[nodiscard]] size_t pixelSizeBytes() const noexcept {
            return m_pixelSizeBytes;
        }

        [[nodiscard]] size_t pitch() const noexcept {
            return m_pitch;
        }

        [[nodiscard]] InteropMode mode() const noexcept {
            return m_mode;
        }

        [[nodiscard]] bool isMapped() const noexcept {
            return m_isMapped;
        }

    private:
        Status initDeviceBuffer();
        Status initExternalSurface();
        size_t calculatePixelSize() const;

        Image* m_image = nullptr;
        InteropMode m_mode = InteropMode::DeviceBufferStaging;

        uint32_t m_width = 0;
        uint32_t m_height = 0;
        size_t m_pixelSizeBytes = 4;
        size_t m_pitch = 0;

        void* m_deviceData = nullptr;
        void* m_hostPinnedData = nullptr;

        cudaExternalMemory_t m_extMemory = nullptr;
        cudaMipmappedArray_t m_mipmappedArray = nullptr;
        cudaArray_t m_cudaArray = nullptr;
        cudaSurfaceObject_t m_surfaceObject = 0;

        bool m_isMapped = false;
        bool m_isInitialized = false;
    };

} // namespace FCT::Cuda

#endif // FCT_CUDA_INTEROP_IMAGE_H
