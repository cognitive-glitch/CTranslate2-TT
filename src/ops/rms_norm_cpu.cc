#include "ctranslate2/ops/rms_norm.h"

#include "cpu/kernels.h"

#ifdef CT2_WITH_TENSTORRENT
#  include "tt/utils.h"
#endif

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void RMSNorm::compute(const StorageView& gamma,
                          const StorageView& input,
                          StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      CPU_ISA_DISPATCH((cpu::rms_norm<ISA>(input.data<T>(),
                                           gamma.data<T>(),
                                           output.data<T>(),
                                           batch_size,
                                           depth,
                                           _epsilon,
                                           _use_residual)));
    }

#define DECLARE_IMPL(T)                                                 \
    template void RMSNorm::compute<Device::CPU, T>(const StorageView&,  \
                                                   const StorageView&,  \
                                                   StorageView&) const;

    DECLARE_IMPL(float)

#ifdef CT2_WITH_TENSTORRENT
    template<>
    template <typename T>
    void RMSNorm::compute<Device::TT, T>(const StorageView& gamma,
                                         const StorageView& input,
                                         StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;
      (void)batch_size;

      auto tin = tt::resolve_tensor(input.data<T>(), input.shape(), input.dtype());
      auto tgamma = tt::resolve_tensor(gamma.data<T>(), gamma.shape(), gamma.dtype());
      auto tout = tt::resolve_tensor(output.data<T>(), output.shape(), output.dtype());
      auto result = ttnn::rms_norm(tin, tgamma, _epsilon, _use_residual);
      ttnn::copy(result, tout);
    }

#define DECLARE_TT_IMPL(T)                                              \
    template void                                                       \
    RMSNorm::compute<Device::TT, T>(const StorageView&,                 \
                                    const StorageView&,                 \
                                    StorageView&) const;

    DECLARE_TT_IMPL(float)
    DECLARE_TT_IMPL(float16_t)
    DECLARE_TT_IMPL(bfloat16_t)
#undef DECLARE_TT_IMPL
#endif

  }
}
