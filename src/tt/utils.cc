#include "tt/utils.h"

#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>

#include "ctranslate2/utils.h"

namespace ctranslate2 {
  namespace tt {
    namespace {
      std::map<uint64_t, Allocation> registry;
      std::mutex registry_mutex;

      Allocation& find_allocation(uint64_t addr) {
        auto it = registry.upper_bound(addr);
        if (it == registry.begin())
          throw std::runtime_error("TT allocation not found for address");
        --it;
        return it->second;
      }

      ttnn::Layout resolve_layout(const Shape& shape, DataType dtype) {
        if (is_float_type(dtype) && is_tile_compatible(shape))
          return ttnn::Layout::TILE;
        return ttnn::Layout::ROW_MAJOR;
      }

      ttnn::Tensor make_tensor_view(const Allocation& allocation,
                                    const Shape& shape,
                                    DataType dtype,
                                    size_t offset_bytes) {
        const size_t element_size = dtype_size(dtype);
        if (offset_bytes % element_size != 0)
          throw std::runtime_error("Unaligned tensor offset for TT backend");

        const dim_t num_elements = compute_size(shape);
        const size_t offset_elements = offset_bytes / element_size;
        const size_t total_elements = allocation.size_bytes / element_size;
        if ((offset_elements + static_cast<size_t>(num_elements)) * element_size > allocation.size_bytes)
          throw std::runtime_error("TT tensor view exceeds allocation size");

        const Shape flat_shape = {static_cast<dim_t>(total_elements)};
        ttnn::Tensor flat_view(allocation.tensor.buffer(),
                               to_tt_shape(flat_shape),
                               to_tt_dtype(dtype),
                               ttnn::Layout::ROW_MAJOR,
                               allocation.tensor.device());

        if (offset_elements != 0 || static_cast<size_t>(num_elements) != total_elements) {
          flat_view = ttnn::slice(flat_view,
                                  {static_cast<int64_t>(offset_elements)},
                                  {static_cast<int64_t>(offset_elements + num_elements)});
        }

        auto reshaped = ttnn::reshape(flat_view, to_tt_shape(shape));
        const auto layout = resolve_layout(shape, dtype);
        if (reshaped.layout() == layout)
          return reshaped;
        return ttnn::Tensor(reshaped.buffer(),
                            to_tt_shape(shape),
                            to_tt_dtype(dtype),
                            layout,
                            reshaped.device());
      }

    }

    void register_allocation(uint64_t addr, Allocation allocation) {
      std::lock_guard<std::mutex> lock(registry_mutex);
      registry.emplace(addr, std::move(allocation));
    }

    void unregister_allocation(uint64_t addr) {
      std::lock_guard<std::mutex> lock(registry_mutex);
      registry.erase(addr);
    }

    ttnn::Tensor resolve_tensor_with_offset(const void* ptr,
                                            const Shape& shape,
                                            DataType dtype,
                                            size_t& offset_bytes) {
      const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
      std::lock_guard<std::mutex> lock(registry_mutex);
      Allocation& allocation = find_allocation(addr);
      const uint64_t base_addr = allocation.tensor.buffer()->address();
      offset_bytes = addr - base_addr;
      return make_tensor_view(allocation, shape, dtype, offset_bytes);
    }

    ttnn::Tensor resolve_tensor(const void* ptr, const Shape& shape, DataType dtype) {
      size_t offset_bytes = 0;
      return resolve_tensor_with_offset(ptr, shape, dtype, offset_bytes);
    }

    void copy_host_to_device(const void* src,
                             void* dst,
                             const Shape& shape,
                             DataType dtype) {
      auto dst_tensor = resolve_tensor(dst, shape, dtype);
      ttnn::Tensor host_tensor = ttnn::from_buffer(const_cast<void*>(src),
                                                   to_tt_shape(shape),
                                                   to_tt_dtype(dtype),
                                                   ttnn::Layout::ROW_MAJOR);
      const auto layout = resolve_layout(shape, dtype);
      if (layout == ttnn::Layout::TILE)
        host_tensor = ttnn::to_layout(host_tensor, layout);
      ttnn::copy(host_tensor, dst_tensor);
    }

    void copy_device_to_host(const void* src,
                             void* dst,
                             const Shape& shape,
                             DataType dtype) {
      auto src_tensor = resolve_tensor(src, shape, dtype);
      if (src_tensor.layout() == ttnn::Layout::TILE)
        src_tensor = ttnn::to_layout(src_tensor, ttnn::Layout::ROW_MAJOR);
      auto host_tensor = ttnn::from_device(src_tensor);
      std::memcpy(dst, host_tensor.buffer()->data(), compute_size(shape) * dtype_size(dtype));
    }

  }
}
