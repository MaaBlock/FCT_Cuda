#include "CudaContext.h"
#include <vector>

namespace FCT::Cuda {

    CudaContext& CudaContext::instance() {
        static CudaContext s_instance;
        return s_instance;
    }

    CudaContext::~CudaContext() {
        if (m_initialized) {
            cudaDeviceReset();
        }
    }

    Status CudaContext::initialize(int deviceId) {
        if (m_initialized) {
            if (m_currentDeviceId == deviceId) {
                return OkStatus();
            }
            shutdown();
        }

        int deviceCount = 0;
        FCT_CHECK_STATUS(cudaGetDeviceCount(&deviceCount));
        if (deviceCount <= 0) {
            return NotFoundError("No CUDA-capable devices detected on current system!");
        }

        if (deviceId < 0 || deviceId >= deviceCount) {
            return InvalidArgumentError("Requested CUDA device index is out of valid range!");
        }

        FCT_CHECK_STATUS(cudaSetDevice(deviceId));

        cudaDeviceProp prop{};
        FCT_CHECK_STATUS(cudaGetDeviceProperties(&prop, deviceId));

        m_deviceInfo.deviceId = deviceId;
        m_deviceInfo.name = prop.name;
        m_deviceInfo.totalGlobalMem = prop.totalGlobalMem;
        m_deviceInfo.multiProcessorCount = prop.multiProcessorCount;
        m_deviceInfo.maxThreadsPerBlock = prop.maxThreadsPerBlock;
        m_deviceInfo.computeCapabilityMajor = prop.major;
        m_deviceInfo.computeCapabilityMinor = prop.minor;

        int timelineSupported = 0;
        cudaDeviceGetAttribute(&timelineSupported, cudaDevAttrTimelineSemaphoreInteropSupported, deviceId);
        m_deviceInfo.supportsTimelineSemaphores = (timelineSupported != 0);

        m_deviceInfo.supportsExternalMemory = (prop.major >= 6);

        m_currentDeviceId = deviceId;
        m_initialized = true;

        return OkStatus();
    }

    Status CudaContext::shutdown() {
        if (!m_initialized) {
            return OkStatus();
        }

        FCT_CHECK_STATUS(cudaDeviceSynchronize());
        FCT_CHECK_STATUS(cudaDeviceReset());

        m_initialized = false;
        m_currentDeviceId = -1;
        m_deviceInfo = {};

        return OkStatus();
    }

    Status CudaContext::synchronize() {
        if (!m_initialized) {
            return FailedPreconditionError("CudaContext is not initialized!");
        }
        FCT_CHECK_STATUS(cudaDeviceSynchronize());
        return OkStatus();
    }

} // namespace FCT::Cuda
