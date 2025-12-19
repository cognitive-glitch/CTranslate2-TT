#include "ctranslate2/ops/conv1d.h"

#include <memory>

#ifdef CT2_WITH_TENSTORRENT
#  include "tt/utils.h"
#endif

namespace ctranslate2 {
  namespace ops {

#ifdef CT2_WITH_TENSTORRENT
    template<>
    template <typename T>
    void Conv1D::compute<Device::TT, T>(const StorageView& input,
                                        const StorageView& weight,
                                        const StorageView* bias,
                                        StorageView& output,
                                        const StorageView* qscale) const {
      StorageView input_cpu = input.to(Device::CPU).to_float32();
      StorageView weight_cpu = weight.to(Device::CPU).to_float32();
      StorageView output_cpu(output.shape(), DataType::FLOAT32, Device::CPU);

      std::unique_ptr<StorageView> bias_cpu;
      if (bias)
        bias_cpu = std::make_unique<StorageView>(bias->to(Device::CPU).to_float32());

      std::unique_ptr<StorageView> qscale_cpu;
      if (qscale)
        qscale_cpu = std::make_unique<StorageView>(qscale->to(Device::CPU).to_float32());

      Conv1D::compute<Device::CPU, float>(input_cpu,
                                          weight_cpu,
                                          bias_cpu ? bias_cpu.get() : nullptr,
                                          output_cpu,
                                          qscale_cpu ? qscale_cpu.get() : nullptr);
      StorageView converted = output_cpu.to(output.dtype());
      output.copy_from(converted);
    }

#define DECLARE_TT_IMPL(T)                                      \
    template void                                               \
    Conv1D::compute<Device::TT, T>(const StorageView& input,    \
                                   const StorageView& weight,   \
                                   const StorageView* bias,     \
                                   StorageView& output,         \
                                   const StorageView* qscale) const;

    DECLARE_TT_IMPL(float)
    DECLARE_TT_IMPL(float16_t)
    DECLARE_TT_IMPL(bfloat16_t)
#undef DECLARE_TT_IMPL
#endif

  }
}
