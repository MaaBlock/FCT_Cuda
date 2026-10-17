#include "CudaInteropImage.h"
#include <cstring>

namespace FCT::Cuda {

    CudaInteropImage::CudaInteropImage(Image* image)
        : m_image(image) {
        if (m_image) {
            m_width = static_cast<uint32_t>(m_image->width());
            m_height = static_cast<uint32_t>(m_image->height());
            m_pixelSizeBytes = calculatePixelSize();
        }
    }

    CudaInteropImage::~CudaInteropImage() {
        release();
    }

    size_t CudaInteropImage::calculatePixelSize() const {
        if (!m_image) {
            return 4;
        }

        Format fmt = m_image->format();
        switch (fmt) {
            case Format::R8G8B8A8_UNORM:
            case Format::B8G8R8A8_UNORM:
            case Format::R8G8B8A8_SRGB:
            case Format::B8G8R8A8_SRGB:
                return 4;
            case Format::R16G16B16A16_SFLOAT:
                return 8;
            case Format::R32G32B32A32_SFLOAT:
                return 16;
            case Format::R32_SFLOAT:
                return 4;
            default:
                return 4;
        }
    }

    Status CudaInteropImage::initialize(InteropMode mode) {
        if (!m_image) {
            return InvalidArgumentError("Cannot initialize CudaInteropImage with null FCT::Image!");
        }

        if (m_width == 0 || m_height == 0) {
            m_width = static_cast<uint32_t>(m_image->width());
            m_height = static_cast<uint32_t>(m_image->height());
        }

        if (m_width == 0 || m_height == 0) {
            return InvalidArgumentError("Image dimensions must be non-zero for CUDA interop!");
        }

        m_pixelSizeBytes = calculatePixelSize();
        m_mode = mode;

        if (m_mode == InteropMode::DirectSurface) {
            Status status = initExternalSurface();
            if (status.ok()) {
                m_isInitialized = true;
                return OkStatus();
            }
            m_mode = InteropMode::DeviceBufferStaging;
        }

        FCT_CHECK_STATUS(initDeviceBuffer());

        m_isInitialized = true;
        return OkStatus();
    }

    Status CudaInteropImage::initDeviceBuffer() {
        size_t rowBytes = static_cast<size_t>(m_width) * m_pixelSizeBytes;
        FCT_CHECK_STATUS(cudaMallocPitch(&m_deviceData, &m_pitch, rowBytes, m_height));

        size_t totalBytes = rowBytes * m_height;
        FCT_CHECK_STATUS(cudaMallocHost(&m_hostPinnedData, totalBytes));

        return OkStatus();
    }

    Status CudaInteropImage::initExternalSurface() {
        return UnavailableError("Direct Vulkan external memory handle export is not supported on current backend!");
    }

    Status CudaInteropImage::release() {
        if (m_surfaceObject != 0) {
            cudaDestroySurfaceObject(m_surfaceObject);
            m_surfaceObject = 0;
        }

        if (m_mipmappedArray) {
            cudaFreeMipmappedArray(m_mipmappedArray);
            m_mipmappedArray = nullptr;
            m_cudaArray = nullptr;
        }

        if (m_extMemory) {
            cudaDestroyExternalMemory(m_extMemory);
            m_extMemory = nullptr;
        }

        if (m_deviceData) {
            cudaFree(m_deviceData);
            m_deviceData = nullptr;
        }

        if (m_hostPinnedData) {
            cudaFreeHost(m_hostPinnedData);
            m_hostPinnedData = nullptr;
        }

        m_isMapped = false;
        m_isInitialized = false;
        return OkStatus();
    }

    StatusOr<cudaSurfaceObject_t> CudaInteropImage::acquireSurface(cudaStream_t stream) {
        if (!m_isInitialized) {
            FCT_CHECK_STATUS(initialize(m_mode));
        }

        if (m_mode == InteropMode::DirectSurface && m_surfaceObject != 0) {
            m_isMapped = true;
            return m_surfaceObject;
        }

        return FailedPreconditionError("Surface object is only available in DirectSurface mode!");
    }

    Status CudaInteropImage::releaseSurface(cudaStream_t stream) {
        if (!m_isMapped) {
            return OkStatus();
        }

        if (stream) {
            FCT_CHECK_STATUS(cudaStreamSynchronize(stream));
        }

        m_isMapped = false;
        return OkStatus();
    }

    StatusOr<void*> CudaInteropImage::acquireDeviceBuffer(size_t* outPitch) {
        if (!m_isInitialized) {
            FCT_CHECK_STATUS(initialize(m_mode));
        }

        if (!m_deviceData) {
            return InternalError("Device buffer memory is unallocated!");
        }

        if (outPitch) {
            *outPitch = m_pitch;
        }

        m_isMapped = true;
        return m_deviceData;
    }

    Status CudaInteropImage::releaseDeviceBuffer(cudaStream_t stream) {
        if (!m_isMapped) {
            return OkStatus();
        }

        if (stream) {
            FCT_CHECK_STATUS(cudaStreamSynchronize(stream));
        }

        m_isMapped = false;
        return OkStatus();
    }

    Status CudaInteropImage::copyToFctImage(cudaStream_t stream) {
        if (!m_deviceData || !m_image) {
            return FailedPreconditionError("Cannot copy unallocated buffer to FCT::Image!");
        }

        size_t rowBytes = static_cast<size_t>(m_width) * m_pixelSizeBytes;

        if (stream) {
            FCT_CHECK_STATUS(cudaMemcpy2DAsync(
                m_hostPinnedData,
                rowBytes,
                m_deviceData,
                m_pitch,
                rowBytes,
                m_height,
                cudaMemcpyDeviceToHost,
                stream));
            FCT_CHECK_STATUS(cudaStreamSynchronize(stream));
        } else {
            FCT_CHECK_STATUS(cudaMemcpy2D(
                m_hostPinnedData,
                rowBytes,
                m_deviceData,
                m_pitch,
                rowBytes,
                m_height,
                cudaMemcpyDeviceToHost));
        }

        size_t totalBytes = rowBytes * m_height;
        m_image->updateData(m_hostPinnedData, totalBytes);

        static int s_loggedFrames = 0;
        if (s_loggedFrames < 5) {
            const uint8_t* p = reinterpret_cast<const uint8_t*>(m_hostPinnedData);
            std::cout << "[CudaInteropImage] Frame " << s_loggedFrames + 1
                      << " transferred to FCT::Image (" << m_width << "x" << m_height
                      << ", " << totalBytes << " bytes, sample byte0=" << static_cast<int>(p[0])
                      << ", byte1=" << static_cast<int>(p[1]) << ")\n";
            s_loggedFrames++;
        }

        return OkStatus();
    }

    Status CudaInteropImage::copyFromFctImage(
        const void* srcData,
        size_t dataSize,
        cudaStream_t stream) {
        if (!m_deviceData) {
            return FailedPreconditionError("Device buffer unallocated!");
        }

        if (!srcData) {
            return InvalidArgumentError("Source data pointer is null!");
        }

        size_t rowBytes = static_cast<size_t>(m_width) * m_pixelSizeBytes;
        size_t totalBytes = rowBytes * m_height;

        if (dataSize < totalBytes) {
            return InvalidArgumentError("Source data size is smaller than image dimension bytes!");
        }

        std::memcpy(m_hostPinnedData, srcData, totalBytes);

        if (stream) {
            FCT_CHECK_STATUS(cudaMemcpy2DAsync(
                m_deviceData,
                m_pitch,
                m_hostPinnedData,
                rowBytes,
                rowBytes,
                m_height,
                cudaMemcpyHostToDevice,
                stream));
        } else {
            FCT_CHECK_STATUS(cudaMemcpy2D(
                m_deviceData,
                m_pitch,
                m_hostPinnedData,
                rowBytes,
                rowBytes,
                m_height,
                cudaMemcpyHostToDevice));
        }

        return OkStatus();
    }

} // namespace FCT::Cuda
