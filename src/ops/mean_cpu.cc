#include "ctranslate2/ops/mean.h"

#include "cpu/parallel.h"
#include "type_dispatch.h"

#ifdef CT2_WITH_TENSTORRENT
#  include "tt/utils.h"
#endif

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Mean::compute(const StorageView& input,
                       const dim_t outer_size,
                       const dim_t axis_size,
                       const dim_t inner_size,
                       const bool get_sum,
                       StorageView& output) const {
      const auto* src = input.data<T>();
      auto* dst = output.data<T>();

      cpu::parallel_for(0, outer_size, 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          for (dim_t j = 0; j < inner_size; ++j) {
            float sum = 0.f;
            for (dim_t k = 0; k < axis_size; ++k) {
              sum += src[i * axis_size * inner_size + k * inner_size + j];
            }
            dst[i * inner_size + j] = sum;
            if (!get_sum)
              dst[i * inner_size + j] /= float(axis_size);
          }
        }
      });
    }

#define DECLARE_IMPL(T)                                         \
    template void                                               \
    Mean::compute<Device::CPU, T>(const StorageView& input,     \
                                  const dim_t outer_size,       \
                                  const dim_t axis_size,        \
                                  const dim_t inner_size,       \
                                  const bool get_sum,           \
                                  StorageView& output) const;

    DECLARE_IMPL(float)

#ifdef CT2_WITH_TENSTORRENT
    template<>
    template <typename T>
    void Mean::compute<Device::TT, T>(const StorageView& input,
                                      const dim_t outer_size,
                                      const dim_t axis_size,
                                      const dim_t inner_size,
                                      const bool get_sum,
                                      StorageView& output) const {
      StorageView input_cpu = input.to(Device::CPU).to_float32();
      StorageView output_cpu(output.shape(), DataType::FLOAT32, Device::CPU);
      Mean::compute<Device::CPU, float>(input_cpu,
                                        outer_size,
                                        axis_size,
                                        inner_size,
                                        get_sum,
                                        output_cpu);
      StorageView converted = output_cpu.to(output.dtype());
      output.copy_from(converted);
    }

#define DECLARE_TT_IMPL(T)                                      \
    template void                                               \
    Mean::compute<Device::TT, T>(const StorageView& input,      \
                                 const dim_t outer_size,        \
                                 const dim_t axis_size,         \
                                 const dim_t inner_size,        \
                                 const bool get_sum,            \
                                 StorageView& output) const;

    DECLARE_TT_IMPL(float)
    DECLARE_TT_IMPL(float16_t)
    DECLARE_TT_IMPL(bfloat16_t)
#undef DECLARE_TT_IMPL
#endif

  }
}
