#include "CudaPass.h"

namespace FCT::Cuda {

    CudaExecutionContext::CudaExecutionContext(
        cudaStream_t stream,
        UnorderedMap<std::string, std::shared_ptr<CudaInteropImage>>& inputs,
        UnorderedMap<std::string, std::shared_ptr<CudaInteropImage>>& outputs)
        : m_stream(stream)
        , m_inputs(inputs)
        , m_outputs(outputs) {}

    StatusOr<CudaInteropImage*> CudaExecutionContext::getImage(const std::string& name) {
        auto inIt = m_inputs.find(name);
        if (inIt != m_inputs.end()) {
            return inIt->second.get();
        }

        auto outIt = m_outputs.find(name);
        if (outIt != m_outputs.end()) {
            return outIt->second.get();
        }

        return NotFoundError("Requested resource name was not declared in CudaPass inputs or outputs: " + name);
    }

    StatusOr<cudaSurfaceObject_t> CudaExecutionContext::getSurface(const std::string& name) {
        auto imgStatus = getImage(name);
        FCT_CHECK_STATUS(imgStatus);
        return imgStatus.value()->acquireSurface(m_stream);
    }

    StatusOr<void*> CudaExecutionContext::getDeviceBuffer(const std::string& name, size_t* outPitch) {
        auto imgStatus = getImage(name);
        FCT_CHECK_STATUS(imgStatus);
        return imgStatus.value()->acquireDeviceBuffer(outPitch);
    }

    CudaPass::CudaPass(std::string name)
        : m_name(std::move(name)) {}

    void CudaPass::addInputImage(
        const std::string& name,
        Image* image) {
        if (!image) {
            return;
        }
        m_inputs[name] = std::make_shared<CudaInteropImage>(image);
    }

    void CudaPass::addOutputImage(
        const std::string& name,
        Image* image) {
        if (!image) {
            return;
        }
        m_outputs[name] = std::make_shared<CudaInteropImage>(image);
    }

    Status CudaPass::dispatch(CudaStream& stream) {
        ZoneScoped();
        ZoneName(m_name.c_str(), m_name.size());
        cudaStream_t s = stream.get();

        for (auto& [name, interopImg] : m_inputs) {
            FCT_CHECK_STATUS(interopImg->initialize());
        }

        for (auto& [name, interopImg] : m_outputs) {
            FCT_CHECK_STATUS(interopImg->initialize());
        }

        CudaExecutionContext ctx(s, m_inputs, m_outputs);
        FCT_CHECK_STATUS(execute(ctx));

        for (auto& [name, interopImg] : m_outputs) {
            FCT_CHECK_STATUS(interopImg->copyToFctImage(s));
            interopImg->releaseDeviceBuffer(s);
        }

        for (auto& [name, interopImg] : m_inputs) {
            interopImg->releaseDeviceBuffer(s);
        }

        return OkStatus();
    }

    Status CudaPass::dispatch(ComputeExecutionContext& context) {
        return dispatch(m_defaultStream);
    }

} // namespace FCT::Cuda
