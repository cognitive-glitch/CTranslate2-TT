#pragma once

#include <cstddef>
#include <cstdint>

#include "tt/ttnn_utils.h"

namespace ctranslate2 {
  namespace tt {

    struct Allocation {
      ttnn::Tensor tensor;
      size_t size_bytes = 0;
    };

    void register_allocation(uint64_t addr, Allocation allocation);
    void unregister_allocation(uint64_t addr);

    ttnn::Tensor resolve_tensor(const void* ptr, const Shape& shape, DataType dtype);
    ttnn::Tensor resolve_tensor_with_offset(const void* ptr,
                                            const Shape& shape,
                                            DataType dtype,
                                            size_t& offset_bytes);

    void copy_host_to_device(const void* src,
                             void* dst,
                             const Shape& shape,
                             DataType dtype);
    void copy_device_to_host(const void* src,
                             void* dst,
                             const Shape& shape,
                             DataType dtype);

  }
}
