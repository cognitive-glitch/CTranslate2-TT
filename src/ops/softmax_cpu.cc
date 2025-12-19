#include "ctranslate2/ops/softmax.h"

#include "cpu/kernels.h"

#ifdef CT2_WITH_TENSTORRENT
#  include "tt/utils.h"
#endif

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void SoftMax::compute(const StorageView& input,
                          const StorageView* lengths,
                          StorageView& output) const {
      const dim_t depth = input.dim(-1);
      const dim_t batch_size = input.size() / depth;

      CPU_ISA_DISPATCH((cpu::softmax<ISA>(input.data<T>(),
                                          lengths ? lengths->data<int32_t>() : nullptr,
                                          output.data<T>(),
                                          batch_size,
                                          depth,
                                          _log)));
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    SoftMax::compute<Device::CPU, T>(const StorageView& input,          \
                                     const StorageView* lengths,        \
                                     StorageView& output) const;

    DECLARE_IMPL(float)

#ifdef CT2_WITH_TENSTORRENT
    template<>
    template <typename T>
    void SoftMax::compute<Device::TT, T>(const StorageView& input,
                                         const StorageView* lengths,
                                         StorageView& output) const {
      if (lengths) {
        StorageView input_cpu = input.to(Device::CPU).to_float32();
        StorageView lengths_cpu = lengths->to(Device::CPU);
        StorageView output_cpu(output.shape(), DataType::FLOAT32, Device::CPU);
        SoftMax::compute<Device::CPU, float>(input_cpu, &lengths_cpu, output_cpu);
        StorageView converted = output_cpu.to(output.dtype());
        output.copy_from(converted);
        return;
      }

      auto tin = tt::resolve_tensor(input.data<T>(), input.shape(), input.dtype());
      auto tout = tt::resolve_tensor(output.data<T>(), output.shape(), output.dtype());
      auto result = _log ? ttnn::log_softmax(tin, -1) : ttnn::softmax(tin, -1);
      ttnn::copy(result, tout);
    }

#define DECLARE_TT_IMPL(T)                                              \
    template void                                                       \
    SoftMax::compute<Device::TT, T>(const StorageView& input,           \
                                    const StorageView* lengths,         \
                                    StorageView& output) const;

    DECLARE_TT_IMPL(float)
    DECLARE_TT_IMPL(float16_t)
    DECLARE_TT_IMPL(bfloat16_t)
#undef DECLARE_TT_IMPL
#endif

  }
}
