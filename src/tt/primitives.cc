#include "ctranslate2/primitives.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <vector>

#include "ctranslate2/utils.h"
#include "type_dispatch.h"
#include "tt/backend.h"
#include "tt/utils.h"

namespace ctranslate2 {
  namespace {

    template <typename T>
    DataType to_data_type() {
      return DataTypeToEnum<T>::value;
    }

    template <typename T>
    std::vector<T> copy_device_to_host_vector(const T* src, dim_t size) {
      std::vector<T> host(size);
      const Shape shape = {size};
      tt::copy_device_to_host(src, host.data(), shape, to_data_type<T>());
      return host;
    }

    template <typename T>
    void copy_host_vector_to_device(const std::vector<T>& host, T* dst) {
      const Shape shape = {static_cast<dim_t>(host.size())};
      tt::copy_host_to_device(host.data(), dst, shape, to_data_type<T>());
    }

    template <typename T>
    void binary_elementwise(const T* a, const T* b, T* c, dim_t size,
                            const std::function<ttnn::Tensor(const ttnn::Tensor&, const ttnn::Tensor&)>& op) {
      const Shape shape = {size};
      auto ta = tt::resolve_tensor(a, shape, to_data_type<T>());
      auto tb = tt::resolve_tensor(b, shape, to_data_type<T>());
      auto tc = tt::resolve_tensor(c, shape, to_data_type<T>());
      auto out = op(ta, tb);
      ttnn::copy(out, tc);
    }

    template <typename T>
    void unary_elementwise(const T* x, T* y, dim_t size,
                           const std::function<ttnn::Tensor(const ttnn::Tensor&)>& op) {
      const Shape shape = {size};
      auto tx = tt::resolve_tensor(x, shape, to_data_type<T>());
      auto ty = tt::resolve_tensor(y, shape, to_data_type<T>());
      auto out = op(tx);
      ttnn::copy(out, ty);
    }

  }

  template<>
  template <typename T>
  T primitives<Device::TT>::at(const T* x, dim_t index) {
    T value{};
    const Shape shape = {1};
    tt::copy_device_to_host(x + index, &value, shape, to_data_type<T>());
    return value;
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::fill(T* x, T a, dim_t size) {
    const Shape shape = {size};
    auto tx = tt::resolve_tensor(x, shape, to_data_type<T>());
    auto filled = ttnn::full(tt::to_tt_shape(shape),
                             a,
                             tt::to_tt_dtype(to_data_type<T>()),
                             tx.layout(),
                             tx.device());
    ttnn::copy(filled, tx);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    auto host = copy_device_to_host_vector(x, size * inc_x);
    for (dim_t i = 0; i < size; ++i)
      host[i * inc_x] = a;
    copy_host_vector_to_device(host, x);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::indexed_fill(T* x, T a, const int32_t* indices, dim_t num_indices) {
    auto host_indices = copy_device_to_host_vector(indices, num_indices);
    int32_t max_index = 0;
    if (!host_indices.empty()) {
      max_index = *std::max_element(host_indices.begin(), host_indices.end());
    }
    auto host = copy_device_to_host_vector(x, static_cast<dim_t>(max_index + 1));
    for (dim_t i = 0; i < num_indices; ++i) {
      const int32_t index = host_indices[i];
      if (index >= 0 && index < static_cast<int32_t>(host.size()))
        host[index] = a;
    }
    copy_host_vector_to_device(host, x);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::copy(const T* x, T* y, dim_t size) {
    const Shape shape = {size};
    auto tx = tt::resolve_tensor(x, shape, to_data_type<T>());
    auto ty = tt::resolve_tensor(y, shape, to_data_type<T>());
    ttnn::copy(tx, ty);
  }

  template<>
  template <typename U, typename V>
  void primitives<Device::TT>::convert(const U* x, V* y, dim_t size) {
    const Shape shape = {size};
    auto tx = tt::resolve_tensor(x, shape, to_data_type<U>());
    auto ty = tt::resolve_tensor(y, shape, to_data_type<V>());
    auto casted = ttnn::typecast(tx, tt::to_tt_dtype(to_data_type<V>()));
    ttnn::copy(casted, ty);
  }

  template<>
  template <typename T>
  T primitives<Device::TT>::sum(const T* array, dim_t size) {
    auto host = copy_device_to_host_vector(array, size);
    return primitives<Device::CPU>::sum(host.data(), size);
  }

  template<>
  template <typename T>
  dim_t primitives<Device::TT>::max_element(const T* array, dim_t size) {
    auto host = copy_device_to_host_vector(array, size);
    return primitives<Device::CPU>::max_element(host.data(), size);
  }

  template<>
  template <typename T>
  T primitives<Device::TT>::max(const T* array, dim_t size) {
    auto host = copy_device_to_host_vector(array, size);
    return primitives<Device::CPU>::max(host.data(), size);
  }

  template<>
  template <typename T>
  T primitives<Device::TT>::amax(const T* array, dim_t size) {
    auto host = copy_device_to_host_vector(array, size);
    return primitives<Device::CPU>::amax(host.data(), size);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::add(T a, const T* x, T* y, dim_t size) {
    const Shape shape = {size};
    auto tx = tt::resolve_tensor(x, shape, to_data_type<T>());
    auto ty = tt::resolve_tensor(y, shape, to_data_type<T>());
    auto scalar = ttnn::full(tt::to_tt_shape(shape),
                             a,
                             tt::to_tt_dtype(to_data_type<T>()),
                             tx.layout(),
                             tx.device());
    auto out = ttnn::add(tx, scalar);
    ttnn::copy(out, ty);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::add(const T* a, const T* b, T* c, dim_t size) {
    binary_elementwise<T>(a, b, c, size, [](const ttnn::Tensor& x, const ttnn::Tensor& y) {
      return ttnn::add(x, y);
    });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::add_batch_broadcast(const T* a, const T* b, T* c,
                                                   dim_t a_size, dim_t b_size) {
    const dim_t batch = b_size / a_size;
    const Shape a_shape = {1, a_size};
    const Shape b_shape = {batch, a_size};
    auto ta = tt::resolve_tensor(a, a_shape, to_data_type<T>());
    auto tb = tt::resolve_tensor(b, b_shape, to_data_type<T>());
    auto tc = tt::resolve_tensor(c, b_shape, to_data_type<T>());
    auto out = ttnn::add(ta, tb);
    ttnn::copy(out, tc);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::add_depth_broadcast(const T* a, const T* b, T* c,
                                                   dim_t a_size, dim_t b_size) {
    const dim_t depth = b_size / a_size;
    const Shape a_shape = {a_size, 1};
    const Shape b_shape = {a_size, depth};
    auto ta = tt::resolve_tensor(a, a_shape, to_data_type<T>());
    auto tb = tt::resolve_tensor(b, b_shape, to_data_type<T>());
    auto tc = tt::resolve_tensor(c, b_shape, to_data_type<T>());
    auto out = ttnn::add(ta, tb);
    ttnn::copy(out, tc);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::sub(const T* a, const T* b, T* c, dim_t size) {
    binary_elementwise<T>(a, b, c, size, [](const ttnn::Tensor& x, const ttnn::Tensor& y) {
      return ttnn::sub(x, y);
    });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::max(T a, const T* x, T* y, dim_t size) {
    const Shape shape = {size};
    auto tx = tt::resolve_tensor(x, shape, to_data_type<T>());
    auto ty = tt::resolve_tensor(y, shape, to_data_type<T>());
    auto scalar = ttnn::full(tt::to_tt_shape(shape),
                             a,
                             tt::to_tt_dtype(to_data_type<T>()),
                             tx.layout(),
                             tx.device());
    auto out = ttnn::maximum(tx, scalar);
    ttnn::copy(out, ty);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::max(const T* a, const T* b, T* c, dim_t size) {
    binary_elementwise<T>(a, b, c, size, [](const ttnn::Tensor& x, const ttnn::Tensor& y) {
      return ttnn::maximum(x, y);
    });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::min(T a, const T* x, T* y, dim_t size) {
    const Shape shape = {size};
    auto tx = tt::resolve_tensor(x, shape, to_data_type<T>());
    auto ty = tt::resolve_tensor(y, shape, to_data_type<T>());
    auto scalar = ttnn::full(tt::to_tt_shape(shape),
                             a,
                             tt::to_tt_dtype(to_data_type<T>()),
                             tx.layout(),
                             tx.device());
    auto out = ttnn::minimum(tx, scalar);
    ttnn::copy(out, ty);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::min(const T* a, const T* b, T* c, dim_t size) {
    binary_elementwise<T>(a, b, c, size, [](const ttnn::Tensor& x, const ttnn::Tensor& y) {
      return ttnn::minimum(x, y);
    });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::mul(T a, const T* x, T* y, dim_t size) {
    const Shape shape = {size};
    auto tx = tt::resolve_tensor(x, shape, to_data_type<T>());
    auto ty = tt::resolve_tensor(y, shape, to_data_type<T>());
    auto scalar = ttnn::full(tt::to_tt_shape(shape),
                             a,
                             tt::to_tt_dtype(to_data_type<T>()),
                             tx.layout(),
                             tx.device());
    auto out = ttnn::mul(tx, scalar);
    ttnn::copy(out, ty);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::mul_batch_broadcast(const T* a, const T* b, T* c,
                                                   dim_t a_size, dim_t b_size) {
    const dim_t batch = b_size / a_size;
    const Shape a_shape = {1, a_size};
    const Shape b_shape = {batch, a_size};
    auto ta = tt::resolve_tensor(a, a_shape, to_data_type<T>());
    auto tb = tt::resolve_tensor(b, b_shape, to_data_type<T>());
    auto tc = tt::resolve_tensor(c, b_shape, to_data_type<T>());
    auto out = ttnn::mul(ta, tb);
    ttnn::copy(out, tc);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::mul(const T* a, const T* b, T* c, dim_t size) {
    binary_elementwise<T>(a, b, c, size, [](const ttnn::Tensor& x, const ttnn::Tensor& y) {
      return ttnn::mul(x, y);
    });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::penalize_previous_tokens(T* scores,
                                                        const T* previous_scores,
                                                        const int32_t* previous_ids,
                                                        T penalty,
                                                        dim_t batch_size,
                                                        dim_t length,
                                                        dim_t vocabulary_size) {
    auto host_scores = copy_device_to_host_vector(scores, batch_size * vocabulary_size);
    auto host_previous = copy_device_to_host_vector(previous_scores, batch_size * length);
    auto host_ids = copy_device_to_host_vector(previous_ids, batch_size * length);
    primitives<Device::CPU>::penalize_previous_tokens(host_scores.data(),
                                                      host_previous.data(),
                                                      host_ids.data(),
                                                      penalty,
                                                      batch_size,
                                                      length,
                                                      vocabulary_size);
    copy_host_vector_to_device(host_scores, scores);
  }

  template<>
  void primitives<Device::TT>::prepare_length_mask(const int32_t* lengths,
                                                   dim_t batch_size,
                                                   dim_t num_heads,
                                                   dim_t num_queries,
                                                   bool mask_future,
                                                   bool multi_query,
                                                   int32_t* mask) {
    const dim_t mask_size = batch_size * num_heads * num_queries;
    auto host_lengths = copy_device_to_host_vector(lengths, batch_size);
    std::vector<int32_t> host_mask(mask_size);
    primitives<Device::CPU>::prepare_length_mask(host_lengths.data(),
                                                 batch_size,
                                                 num_heads,
                                                 num_queries,
                                                 mask_future,
                                                 multi_query,
                                                 host_mask.data());
    copy_host_vector_to_device(host_mask, mask);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::transpose_2d(const T* a, const dim_t* dims, T* b) {
    const Shape a_shape = {dims[0], dims[1]};
    const Shape b_shape = {dims[1], dims[0]};
    auto ta = tt::resolve_tensor(a, a_shape, to_data_type<T>());
    auto tb = tt::resolve_tensor(b, b_shape, to_data_type<T>());
    auto out = ttnn::transpose(ta, 0, 1);
    ttnn::copy(out, tb);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::transpose_3d(const T* a, const dim_t* dims, const dim_t* perm, T* b) {
    const Shape a_shape = {dims[0], dims[1], dims[2]};
    const Shape b_shape = {dims[perm[0]], dims[perm[1]], dims[perm[2]]};
    auto ta = tt::resolve_tensor(a, a_shape, to_data_type<T>());
    auto tb = tt::resolve_tensor(b, b_shape, to_data_type<T>());
    std::vector<int64_t> perm64 = {perm[0], perm[1], perm[2]};
    auto out = ttnn::permute(ta, perm64);
    ttnn::copy(out, tb);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::transpose_4d(const T* a, const dim_t* dims, const dim_t* perm, T* b) {
    const Shape a_shape = {dims[0], dims[1], dims[2], dims[3]};
    const Shape b_shape = {dims[perm[0]], dims[perm[1]], dims[perm[2]], dims[perm[3]]};
    auto ta = tt::resolve_tensor(a, a_shape, to_data_type<T>());
    auto tb = tt::resolve_tensor(b, b_shape, to_data_type<T>());
    std::vector<int64_t> perm64 = {perm[0], perm[1], perm[2], perm[3]};
    auto out = ttnn::permute(ta, perm64);
    ttnn::copy(out, tb);
  }

  template<>
  template <typename T>
  float primitives<Device::TT>::logsumexp(const T* x, dim_t size) {
    auto host = copy_device_to_host_vector(x, size);
    return primitives<Device::CPU>::logsumexp(host.data(), size);
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::exp(const T* x, T* y, dim_t size) {
    unary_elementwise<T>(x, y, size, [](const ttnn::Tensor& t) { return ttnn::exp(t); });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::log(const T* x, T* y, dim_t size) {
    unary_elementwise<T>(x, y, size, [](const ttnn::Tensor& t) { return ttnn::log(t); });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::cos(const T* x, T* y, dim_t size) {
    unary_elementwise<T>(x, y, size, [](const ttnn::Tensor& t) { return ttnn::cos(t); });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::sin(const T* x, T* y, dim_t size) {
    unary_elementwise<T>(x, y, size, [](const ttnn::Tensor& t) { return ttnn::sin(t); });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::tanh(const T* x, T* y, dim_t size) {
    unary_elementwise<T>(x, y, size, [](const ttnn::Tensor& t) { return ttnn::tanh(t); });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::relu(const T* x, T* y, dim_t size) {
    unary_elementwise<T>(x, y, size, [](const ttnn::Tensor& t) { return ttnn::relu(t); });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::gelu(const T* x, T* y, dim_t size) {
    unary_elementwise<T>(x, y, size, [](const ttnn::Tensor& t) { return ttnn::gelu(t); });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::gelu_tanh(const T* x, T* y, dim_t size) {
    unary_elementwise<T>(x, y, size, [](const ttnn::Tensor& t) { return ttnn::gelu(t); });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::gelu_sigmoid(const T* x, T* y, dim_t size) {
    unary_elementwise<T>(x, y, size, [](const ttnn::Tensor& t) { return ttnn::gelu(t); });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::sigmoid(const T* x, T* y, dim_t size) {
    unary_elementwise<T>(x, y, size, [](const ttnn::Tensor& t) { return ttnn::sigmoid(t); });
  }

  template<>
  template <typename T>
  void primitives<Device::TT>::swish(const T* x, T* y, dim_t size) {
    unary_elementwise<T>(x, y, size, [](const ttnn::Tensor& t) { return ttnn::silu(t); });
  }

  template<>
  void primitives<Device::TT>::compute_u8_compensation(const int8_t*,
                                                       bool,
                                                       dim_t,
                                                       dim_t,
                                                       float,
                                                       int32_t*) {
    throw std::runtime_error("compute_u8_compensation is not supported on TT");
  }

  template<>
  template <typename T>
  dim_t primitives<Device::TT>::gemm_pack_b(const T*,
                                            const bool,
                                            const dim_t,
                                            const dim_t,
                                            const float,
                                            T*) {
    return 0;
  }

  template<>
  template <typename In, typename Out>
  void primitives<Device::TT>::gemm(bool a_is_packed, bool b_is_packed,
                                    bool transpose_a, bool transpose_b,
                                    dim_t m, dim_t n, dim_t k,
                                    float alpha,
                                    const In* a, dim_t lda,
                                    const In* b, dim_t ldb,
                                    float beta,
                                    Out* c, dim_t ldc,
                                    const Out* a_shift_compensation) {
    if (a_is_packed || b_is_packed)
      throw std::runtime_error("Packed GEMM is not supported on TT");
    if (a_shift_compensation)
      throw std::runtime_error("Shift compensation GEMM is not supported on TT");

    (void)lda;
    (void)ldb;
    (void)ldc;

    Shape a_shape = {m, k};
    Shape b_shape = {k, n};
    Shape c_shape = {m, n};

    auto ta = tt::resolve_tensor(a, a_shape, to_data_type<In>());
    auto tb = tt::resolve_tensor(b, b_shape, to_data_type<In>());
    auto tc = tt::resolve_tensor(c, c_shape, to_data_type<Out>());

    if (transpose_a)
      ta = ttnn::transpose(ta, 0, 1);
    if (transpose_b)
      tb = ttnn::transpose(tb, 0, 1);

    if (ta.layout() != ttnn::Layout::TILE)
      ta = ttnn::to_layout(ta, ttnn::Layout::TILE);
    if (tb.layout() != ttnn::Layout::TILE)
      tb = ttnn::to_layout(tb, ttnn::Layout::TILE);

    auto result = ttnn::matmul(ta, tb);
    if (alpha != 1.f) {
      auto alpha_tensor = ttnn::full(tt::to_tt_shape(c_shape),
                                     alpha,
                                     tt::to_tt_dtype(to_data_type<Out>()),
                                     result.layout(),
                                     result.device());
      result = ttnn::mul(result, alpha_tensor);
    }
    if (beta != 0.f) {
      auto beta_tensor = ttnn::full(tt::to_tt_shape(c_shape),
                                    beta,
                                    tt::to_tt_dtype(to_data_type<Out>()),
                                    result.layout(),
                                    result.device());
      auto scaled_c = ttnn::mul(tc, beta_tensor);
      result = ttnn::add(result, scaled_c);
    }
    ttnn::copy(result, tc);
  }

  template<>
  template <typename In, typename Out>
  void primitives<Device::TT>::gemm_batch_strided(bool transpose_a, bool transpose_b,
                                                  dim_t m, dim_t n, dim_t k,
                                                  float alpha,
                                                  const In* a, dim_t lda, dim_t stridea,
                                                  const In* b, dim_t ldb, dim_t strideb,
                                                  float beta,
                                                  Out* c, dim_t ldc, dim_t stridec,
                                                  dim_t batch_size) {
    for (dim_t i = 0; i < batch_size; ++i) {
      const In* a_i = a + (i * stridea);
      const In* b_i = b + (i * strideb);
      Out* c_i = c + (i * stridec);
      gemm(/*a_is_packed=*/false, /*b_is_packed=*/false,
           transpose_a, transpose_b,
           m, n, k,
           alpha,
           a_i, lda,
           b_i, ldb,
           beta,
           c_i, ldc);
    }
  }

  template<>
  template <typename T>
  void cross_device_primitives<Device::CPU, Device::TT>::copy(const T* x, T* y, dim_t size) {
    const Shape shape = {size};
    tt::copy_host_to_device(x, y, shape, to_data_type<T>());
  }

  template<>
  template <typename T>
  void cross_device_primitives<Device::TT, Device::CPU>::copy(const T* x, T* y, dim_t size) {
    const Shape shape = {size};
    tt::copy_device_to_host(x, y, shape, to_data_type<T>());
  }

#define DECLARE_IMPL(T)                                                 \
  template T                                                            \
  primitives<Device::TT>::at(const T* x, dim_t index);                  \
  template void                                                         \
  primitives<Device::TT>::fill(T* x, T a, dim_t size);                  \
  template void                                                         \
  primitives<Device::TT>::strided_fill(T* x, T a, dim_t inc_x, dim_t size); \
  template void                                                         \
  primitives<Device::TT>::indexed_fill(T*, T, const int32_t*, dim_t);   \
  template void                                                         \
  primitives<Device::TT>::copy(const T* x, T* y, dim_t size);           \
  template T                                                            \
  primitives<Device::TT>::sum(const T* array, dim_t size);              \
  template dim_t                                                        \
  primitives<Device::TT>::max_element(const T* array, dim_t size);      \
  template T                                                            \
  primitives<Device::TT>::max(const T* array, dim_t size);              \
  template void                                                         \
  primitives<Device::TT>::add(T a, const T* x, T* y, dim_t size);       \
  template void                                                         \
  primitives<Device::TT>::add_batch_broadcast(const T* a, const T* b, T* c, \
                                              dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::TT>::add_depth_broadcast(const T* a, const T* b, T* c, \
                                              dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::TT>::min(T a, const T* x, T* y, dim_t size);       \
  template void                                                         \
  primitives<Device::TT>::max(T a, const T* x, T* y, dim_t size);       \
  template void                                                         \
  primitives<Device::TT>::mul_batch_broadcast(const T* a, const T* b, T* c, \
                                              dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::TT>::penalize_previous_tokens(T*,                  \
                                                   const T*,            \
                                                   const int32_t*,      \
                                                   T,                   \
                                                   dim_t,               \
                                                   dim_t,               \
                                                   dim_t);              \
  template void                                                         \
  primitives<Device::TT>::transpose_2d(const T* a,                      \
                                       const dim_t* dims,               \
                                       T* b);                           \
  template void                                                         \
  primitives<Device::TT>::transpose_3d(const T* a,                      \
                                       const dim_t* dims,               \
                                       const dim_t* perm,               \
                                       T* b);                           \
  template void                                                         \
  primitives<Device::TT>::transpose_4d(const T* a,                      \
                                       const dim_t* dims,               \
                                       const dim_t* perm,               \
                                       T* b);                           \
  template void                                                         \
  cross_device_primitives<Device::CPU, Device::TT>::copy<T>(const T*, T*, dim_t); \
  template void                                                         \
  cross_device_primitives<Device::TT, Device::CPU>::copy<T>(const T*, T*, dim_t);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

#define DECLARE_IMPL_FLOAT(T)                                           \
  template T                                                            \
  primitives<Device::TT>::amax(const T* array, dim_t size);             \
  template void                                                         \
  primitives<Device::TT>::add(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::TT>::sub(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::TT>::min(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::TT>::max(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::TT>::mul(const T* a, const T* b, T* c, dim_t size); \
  template float                                                        \
  primitives<Device::TT>::logsumexp(const T* x, dim_t size);            \
  template void                                                         \
  primitives<Device::TT>::exp(const T* x, T* y, dim_t size);            \
  template void                                                         \
  primitives<Device::TT>::log(const T* x, T* y, dim_t size);            \
  template void                                                         \
  primitives<Device::TT>::cos(const T* x, T* y, dim_t size);            \
  template void                                                         \
  primitives<Device::TT>::sin(const T* x, T* y, dim_t size);            \
  template void                                                         \
  primitives<Device::TT>::tanh(const T* x, T* y, dim_t size);           \
  template void                                                         \
  primitives<Device::TT>::relu(const T* x, T* y, dim_t size);           \
  template void                                                         \
  primitives<Device::TT>::gelu(const T* x, T* y, dim_t size);           \
  template void                                                         \
  primitives<Device::TT>::gelu_tanh(const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::TT>::gelu_sigmoid(const T* x, T* y, dim_t size);   \
  template void                                                         \
  primitives<Device::TT>::sigmoid(const T* x, T* y, dim_t size);        \
  template void                                                         \
  primitives<Device::TT>::swish(const T* x, T* y, dim_t size);

  DECLARE_IMPL_FLOAT(float16_t)
  DECLARE_IMPL_FLOAT(bfloat16_t)
  DECLARE_IMPL_FLOAT(float)

  template void primitives<Device::TT>::convert(const float*, float16_t*, dim_t);
  template void primitives<Device::TT>::convert(const float16_t*, float*, dim_t);
  template void primitives<Device::TT>::convert(const float*, bfloat16_t*, dim_t);
  template void primitives<Device::TT>::convert(const bfloat16_t*, float*, dim_t);
  template void primitives<Device::TT>::convert(const float16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::TT>::convert(const bfloat16_t*, float16_t*, dim_t);

  template void primitives<Device::TT>::gemm<float, float>(bool, bool, bool, bool,
                                                           dim_t, dim_t, dim_t,
                                                           float,
                                                           const float*, dim_t,
                                                           const float*, dim_t,
                                                           float,
                                                           float*, dim_t,
                                                           const float*);

  template void primitives<Device::TT>::gemm<float16_t, float16_t>(bool, bool, bool, bool,
                                                                   dim_t, dim_t, dim_t,
                                                                   float,
                                                                   const float16_t*, dim_t,
                                                                   const float16_t*, dim_t,
                                                                   float,
                                                                   float16_t*, dim_t,
                                                                   const float16_t*);

  template void primitives<Device::TT>::gemm<bfloat16_t, bfloat16_t>(bool, bool, bool, bool,
                                                                     dim_t, dim_t, dim_t,
                                                                     float,
                                                                     const bfloat16_t*, dim_t,
                                                                     const bfloat16_t*, dim_t,
                                                                     float,
                                                                     bfloat16_t*, dim_t,
                                                                     const bfloat16_t*);

  template void primitives<Device::TT>::gemm_batch_strided<float, float>(bool, bool,
                                                                         dim_t, dim_t, dim_t,
                                                                         float,
                                                                         const float*, dim_t, dim_t,
                                                                         const float*, dim_t, dim_t,
                                                                         float,
                                                                         float*, dim_t, dim_t,
                                                                         dim_t);

  template void primitives<Device::TT>::gemm_batch_strided<float16_t, float16_t>(bool, bool,
                                                                                 dim_t, dim_t, dim_t,
                                                                                 float,
                                                                                 const float16_t*, dim_t, dim_t,
                                                                                 const float16_t*, dim_t, dim_t,
                                                                                 float,
                                                                                 float16_t*, dim_t, dim_t,
                                                                                 dim_t);

  template void primitives<Device::TT>::gemm_batch_strided<bfloat16_t, bfloat16_t>(bool, bool,
                                                                                   dim_t, dim_t, dim_t,
                                                                                   float,
                                                                                   const bfloat16_t*, dim_t, dim_t,
                                                                                   const bfloat16_t*, dim_t, dim_t,
                                                                                   float,
                                                                                   bfloat16_t*, dim_t, dim_t,
                                                                                   dim_t);

}
