#ifndef FCT_CUDA_CONTEXT_H
#define FCT_CUDA_CONTEXT_H

#include "CudaCommon.h"

namespace FCT::Cuda {

    struct DeviceInfo {
        int deviceId = 0;
        std::string name;
        size_t totalGlobalMem = 0;
        int multiProcessorCount = 0;
        int maxThreadsPerBlock = 0;
        int computeCapabilityMajor = 0;
        int computeCapabilityMinor = 0;
        bool supportsTimelineSemaphores = false;
        bool supportsExternalMemory = false;
    };

    class CudaContext {
    public:
        static CudaContext& instance();

        Status initialize(int deviceId = 0);
        Status shutdown();

        [[nodiscard]] bool isInitialized() const noexcept {
            return m_initialized;
        }

        [[nodiscard]] int currentDeviceId() const noexcept {
            return m_currentDeviceId;
        }

        [[nodiscard]] const DeviceInfo& deviceInfo() const noexcept {
            return m_deviceInfo;
        }

        Status synchronize();

    private:
        CudaContext() = default;
        ~CudaContext();

        bool m_initialized = false;
        int m_currentDeviceId = -1;
        DeviceInfo m_deviceInfo{};
    };

} // namespace FCT::Cuda

#endif // FCT_CUDA_CONTEXT_H
