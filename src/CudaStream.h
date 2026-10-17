#ifndef FCT_CUDA_STREAM_H
#define FCT_CUDA_STREAM_H

#include "CudaCommon.h"

namespace FCT::Cuda {

    class CudaStream {
    public:
        CudaStream();
        explicit CudaStream(unsigned int flags);
        ~CudaStream();

        CudaStream(const CudaStream&) = delete;
        CudaStream& operator=(const CudaStream&) = delete;

        CudaStream(CudaStream&& other) noexcept;
        CudaStream& operator=(CudaStream&& other) noexcept;

        Status initialize(unsigned int flags = cudaStreamNonBlocking);
        Status destroy();
        Status synchronize();

        [[nodiscard]] cudaStream_t get() const noexcept {
            return m_stream;
        }

        [[nodiscard]] bool isValid() const noexcept {
            return m_stream != nullptr;
        }

    private:
        cudaStream_t m_stream = nullptr;
    };

    class CudaEvent {
    public:
        CudaEvent();
        explicit CudaEvent(unsigned int flags);
        ~CudaEvent();

        CudaEvent(const CudaEvent&) = delete;
        CudaEvent& operator=(const CudaEvent&) = delete;

        CudaEvent(CudaEvent&& other) noexcept;
        CudaEvent& operator=(CudaEvent&& other) noexcept;

        Status initialize(unsigned int flags = cudaEventDisableTiming);
        Status record(cudaStream_t stream = nullptr);
        Status wait(cudaStream_t stream = nullptr);
        Status synchronize();
        Status destroy();

        [[nodiscard]] cudaEvent_t get() const noexcept {
            return m_event;
        }

        [[nodiscard]] bool isValid() const noexcept {
            return m_event != nullptr;
        }

    private:
        cudaEvent_t m_event = nullptr;
    };

} // namespace FCT::Cuda

#endif // FCT_CUDA_STREAM_H
