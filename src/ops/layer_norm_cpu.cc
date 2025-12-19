#include "ctranslate2/ops/layer_norm.h"

#include <memory>

#include "cpu/kernels.h"

#ifdef CT2_WITH_TENSTORRENT
#  include "tt/utils.h"
#endif

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void LayerNorm::compute(const StorageView* beta,
                            const StorageView* gamma,
                            const StorageView& input,
                            const dim_t axis,
                            const dim_t outer_size,
                            const dim_t axis_size,
                            const dim_t inner_size,
                            StorageView& output) const {
      if (axis == input.rank() - 1 && beta && gamma) {
        CPU_ISA_DISPATCH((cpu::layer_norm<ISA>(input.data<T>(),
                                               gamma->data<T>(),
                                               beta->data<T>(),
                                               output.data<T>(),
                                               outer_size,
                                               axis_size,
                                               _epsilon)));
      } else {
        CPU_ISA_DISPATCH((cpu::layer_norm_axis<ISA>(input.data<T>(),
                                                    gamma ? gamma->data<T>() : nullptr,
                                                    beta ? beta->data<T>() : nullptr,
                                                    output.data<T>(),
                                                    outer_size,
                                                    axis_size,
                                                    inner_size,
                                                    _epsilon)));
      }
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    LayerNorm::compute<Device::CPU, T>(const StorageView* beta,         \
                                       const StorageView* gamma,        \
                                       const StorageView& input,        \
                                       const dim_t axis,                \
                                       const dim_t outer_size,          \
                                       const dim_t axis_size,           \
                                       const dim_t inner_size,          \
                                       StorageView& output) const;

    DECLARE_IMPL(float)

#ifdef CT2_WITH_TENSTORRENT
    template<>
    template <typename T>
    void LayerNorm::compute<Device::TT, T>(const StorageView* beta,
                                           const StorageView* gamma,
                                           const StorageView& input,
                                           const dim_t axis,
                                           const dim_t outer_size,
                                           const dim_t axis_size,
                                           const dim_t inner_size,
                                           StorageView& output) const {
      (void)outer_size;
      (void)axis_size;
      (void)inner_size;

      if (axis == input.rank() - 1 && beta && gamma) {
        auto tin = tt::resolve_tensor(input.data<T>(), input.shape(), input.dtype());
        auto tgamma = tt::resolve_tensor(gamma->data<T>(), gamma->shape(), gamma->dtype());
        auto tbeta = tt::resolve_tensor(beta->data<T>(), beta->shape(), beta->dtype());
        auto tout = tt::resolve_tensor(output.data<T>(), output.shape(), output.dtype());
        auto result = ttnn::layer_norm(tin, tgamma, tbeta, _epsilon);
        ttnn::copy(result, tout);
        return;
      }

      StorageView input_cpu = input.to(Device::CPU).to_float32();
      StorageView output_cpu(output.shape(), DataType::FLOAT32, Device::CPU);
      std::unique_ptr<StorageView> beta_cpu;
      std::unique_ptr<StorageView> gamma_cpu;
      if (beta)
        beta_cpu = std::make_unique<StorageView>(beta->to(Device::CPU).to_float32());
      if (gamma)
        gamma_cpu = std::make_unique<StorageView>(gamma->to(Device::CPU).to_float32());
      LayerNorm::compute<Device::CPU, float>(beta_cpu ? beta_cpu.get() : nullptr,
                                             gamma_cpu ? gamma_cpu.get() : nullptr,
                                             input_cpu,
                                             axis,
                                             outer_size,
                                             axis_size,
                                             inner_size,
                                             output_cpu);
      StorageView converted = output_cpu.to(output.dtype());
      output.copy_from(converted);
    }

#define DECLARE_TT_IMPL(T)                                              \
    template void                                                       \
    LayerNorm::compute<Device::TT, T>(const StorageView* beta,          \
                                      const StorageView* gamma,         \
                                      const StorageView& input,         \
                                      const dim_t axis,                 \
                                      const dim_t outer_size,           \
                                      const dim_t axis_size,            \
                                      const dim_t inner_size,           \
                                      StorageView& output) const;

    DECLARE_TT_IMPL(float)
    DECLARE_TT_IMPL(float16_t)
    DECLARE_TT_IMPL(bfloat16_t)
#undef DECLARE_TT_IMPL
#endif

  }
}
