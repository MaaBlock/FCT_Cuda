#ifndef FCT_CUDA_COMMON_H
#define FCT_CUDA_COMMON_H

#include <cuda_runtime.h>
#include <string>
#include <string_view>
#include <memory>
#include <cstdint>

#include <FCT/Bases.h>

namespace FCT {

    inline Status ToStatus(cudaError_t error) {
        if (error == cudaSuccess) {
            return OkStatus();
        }
        return InternalError(StrCat("CUDA Error [", cudaGetErrorName(error), "]: ", cudaGetErrorString(error)));
    }

} // namespace FCT

#endif // FCT_CUDA_COMMON_H
