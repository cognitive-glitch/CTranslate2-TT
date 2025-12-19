#include "ctranslate2/ops/flash_attention.h"

#include <stdexcept>

#include "dispatch.h"

#ifdef CT2_WITH_TENSTORRENT
#  include "tt/utils.h"
#endif

namespace ctranslate2 {
  namespace ops {

#ifdef CT2_WITH_TENSTORRENT
    template<>
    void FlashAttention::compute<Device::TT>(StorageView& queries,
                                             StorageView& keys,
                                             StorageView& values,
                                             StorageView& output,
                                             StorageView* cached_keys,
                                             StorageView* cached_values,
                                             StorageView* attention,
                                             bool return_normalized_attention,
                                             StorageView* rotary_cos,
                                             StorageView* rotary_sin,
                                             const bool rotary_interleave,
                                             StorageView* alibi,
                                             dim_t offset) const {
      (void)cached_keys;
      (void)cached_values;
      (void)rotary_interleave;
      (void)offset;

      if (attention || return_normalized_attention || rotary_cos || rotary_sin || alibi)
        throw std::runtime_error("FlashAttention TT backend does not support attention return or rotary/alibi inputs");
      if (queries.dtype() != DataType::FLOAT16 && queries.dtype() != DataType::BFLOAT16)
        throw std::invalid_argument("FlashAttention TT backend only supports float16 or bfloat16");
      if (_sliding_window > 0)
        throw std::runtime_error("FlashAttention TT backend does not support sliding window attention");

      output.resize_as(queries);

      TYPE_DISPATCH(queries.dtype(), {
        auto tq = tt::resolve_tensor(queries.data<T>(), queries.shape(), queries.dtype());
        auto tk = tt::resolve_tensor(keys.data<T>(), keys.shape(), keys.dtype());
        auto tv = tt::resolve_tensor(values.data<T>(), values.shape(), values.dtype());
        auto tout = tt::resolve_tensor(output.data<T>(), output.shape(), output.dtype());
        const bool is_causal = (_sliding_window == 0);
        auto result = ttnn::transformer::scaled_dot_product_attention(tq, tk, tv, is_causal, _queries_scale);
        ttnn::copy(result, tout);
      });
    }
#endif

  }
}
