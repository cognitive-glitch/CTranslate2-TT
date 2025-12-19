#include "ctranslate2/ops/multinomial.h"

#include "ctranslate2/random.h"

#ifdef CT2_WITH_TENSTORRENT
#  include "tt/utils.h"
#endif

namespace ctranslate2 {
  namespace ops {

    template <typename In, typename Out>
    static void multinomial_kernel(const In* input,
                                   dim_t batch_size,
                                   dim_t class_size,
                                   dim_t sample_size,
                                   Out* output) {
      auto& generator = get_random_generator();

      for (dim_t i = 0; i < batch_size; ++i) {
        const In* input_data = input + i * class_size;
        Out* output_data = output + i * sample_size;

        std::discrete_distribution<Out> distribution(input_data, input_data + class_size);
        for (dim_t j = 0; j < sample_size; ++j)
          output_data[j] = distribution(generator);
      }
    }

    template <Device D, typename T>
    void Multinomial::compute(const StorageView& input, StorageView& output) const {
      const dim_t class_size = input.dim(-1);
      const dim_t batch_size = input.size() / class_size;
      multinomial_kernel(input.data<T>(),
                         batch_size,
                         class_size,
                         _sample_size,
                         output.data<int32_t>());
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Multinomial::compute<Device::CPU, T>(const StorageView& input,      \
                                         StorageView& output) const;

    DECLARE_IMPL(float)

#ifdef CT2_WITH_TENSTORRENT
    template<>
    template <typename T>
    void Multinomial::compute<Device::TT, T>(const StorageView& input,
                                             StorageView& output) const {
      StorageView input_cpu = input.to(Device::CPU).to_float32();
      StorageView output_cpu(output.shape(), output.dtype(), Device::CPU);
      Multinomial::compute<Device::CPU, float>(input_cpu, output_cpu);
      output.copy_from(output_cpu);
    }

#define DECLARE_TT_IMPL(T)                                      \
    template void                                               \
    Multinomial::compute<Device::TT, T>(const StorageView& input, \
                                        StorageView& output) const;

    DECLARE_TT_IMPL(float)
    DECLARE_TT_IMPL(float16_t)
    DECLARE_TT_IMPL(bfloat16_t)
#undef DECLARE_TT_IMPL
#endif

  }
}
