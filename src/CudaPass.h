#ifndef FCT_CUDA_PASS_H
#define FCT_CUDA_PASS_H

#include "CudaCommon.h"
#include "CudaStream.h"
#include "CudaInteropImage.h"
#include "Context/Image.h"
#include "Context/ComputePass.h"
#include "MutilThreadBase/RefCount.h"

namespace FCT::Cuda {

    class CudaExecutionContext {
    public:
        CudaExecutionContext(
            cudaStream_t stream,
            UnorderedMap<std::string, std::shared_ptr<CudaInteropImage>>& inputs,
            UnorderedMap<std::string, std::shared_ptr<CudaInteropImage>>& outputs);

        [[nodiscard]] cudaStream_t stream() const noexcept {
            return m_stream;
        }

        StatusOr<cudaSurfaceObject_t> getSurface(const std::string& name);
        StatusOr<void*> getDeviceBuffer(const std::string& name, size_t* outPitch = nullptr);
        StatusOr<CudaInteropImage*> getImage(const std::string& name);

    private:
        cudaStream_t m_stream;
        UnorderedMap<std::string, std::shared_ptr<CudaInteropImage>>& m_inputs;
        UnorderedMap<std::string, std::shared_ptr<CudaInteropImage>>& m_outputs;
    };

    class CudaPass : public IComputeDispatcher {
    public:
        explicit CudaPass(std::string name);
        ~CudaPass() override = default;

        [[nodiscard]] const std::string& name() const noexcept {
            return m_name;
        }

        void addInputImage(
            const std::string& name,
            Image* image);

        void addOutputImage(
            const std::string& name,
            Image* image);

        virtual Status execute(CudaExecutionContext& context) = 0;

        Status dispatch(CudaStream& stream);
        Status dispatch(ComputeExecutionContext& context) override;

    protected:
        std::string m_name;
        CudaStream m_defaultStream;
        UnorderedMap<std::string, std::shared_ptr<CudaInteropImage>> m_inputs;
        UnorderedMap<std::string, std::shared_ptr<CudaInteropImage>> m_outputs;
    };

} // namespace FCT::Cuda

#endif // FCT_CUDA_PASS_H
