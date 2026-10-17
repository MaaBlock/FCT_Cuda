#include "CudaStream.h"
#include <utility>

namespace FCT::Cuda {

    CudaStream::CudaStream() {
        initialize();
    }

    CudaStream::CudaStream(unsigned int flags) {
        initialize(flags);
    }

    CudaStream::~CudaStream() {
        destroy();
    }

    CudaStream::CudaStream(CudaStream&& other) noexcept
        : m_stream(other.m_stream) {
        other.m_stream = nullptr;
    }

    CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            destroy();
            m_stream = other.m_stream;
            other.m_stream = nullptr;
        }
        return *this;
    }

    Status CudaStream::initialize(unsigned int flags) {
        if (m_stream) {
            return OkStatus();
        }
        FCT_CHECK_STATUS(cudaStreamCreateWithFlags(&m_stream, flags));
        return OkStatus();
    }

    Status CudaStream::destroy() {
        if (!m_stream) {
            return OkStatus();
        }
        cudaStreamSynchronize(m_stream);
        cudaStream_t streamToDestroy = m_stream;
        m_stream = nullptr;
        FCT_CHECK_STATUS(cudaStreamDestroy(streamToDestroy));
        return OkStatus();
    }

    Status CudaStream::synchronize() {
        if (!m_stream) {
            return FailedPreconditionError("Cannot synchronize uninitialized CudaStream!");
        }
        FCT_CHECK_STATUS(cudaStreamSynchronize(m_stream));
        return OkStatus();
    }

    CudaEvent::CudaEvent() {
        initialize();
    }

    CudaEvent::CudaEvent(unsigned int flags) {
        initialize(flags);
    }

    CudaEvent::~CudaEvent() {
        destroy();
    }

    CudaEvent::CudaEvent(CudaEvent&& other) noexcept
        : m_event(other.m_event) {
        other.m_event = nullptr;
    }

    CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            destroy();
            m_event = other.m_event;
            other.m_event = nullptr;
        }
        return *this;
    }

    Status CudaEvent::initialize(unsigned int flags) {
        if (m_event) {
            return OkStatus();
        }
        FCT_CHECK_STATUS(cudaEventCreateWithFlags(&m_event, flags));
        return OkStatus();
    }

    Status CudaEvent::record(cudaStream_t stream) {
        if (!m_event) {
            return FailedPreconditionError("Cannot record uninitialized CudaEvent!");
        }
        FCT_CHECK_STATUS(cudaEventRecord(m_event, stream));
        return OkStatus();
    }

    Status CudaEvent::wait(cudaStream_t stream) {
        if (!m_event) {
            return FailedPreconditionError("Cannot wait on uninitialized CudaEvent!");
        }
        FCT_CHECK_STATUS(cudaStreamWaitEvent(stream, m_event, 0));
        return OkStatus();
    }

    Status CudaEvent::synchronize() {
        if (!m_event) {
            return FailedPreconditionError("Cannot synchronize uninitialized CudaEvent!");
        }
        FCT_CHECK_STATUS(cudaEventSynchronize(m_event));
        return OkStatus();
    }

    Status CudaEvent::destroy() {
        if (!m_event) {
            return OkStatus();
        }
        cudaEvent_t eventToDestroy = m_event;
        m_event = nullptr;
        FCT_CHECK_STATUS(cudaEventDestroy(eventToDestroy));
        return OkStatus();
    }

} // namespace FCT::Cuda
