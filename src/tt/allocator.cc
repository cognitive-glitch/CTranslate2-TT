#include "ctranslate2/allocator.h"

#include "tt/backend.h"
#include "tt/utils.h"

namespace ctranslate2 {
  namespace tt {

    class TTAllocator : public Allocator {
    public:
      void* allocate(size_t size, int device_index) override {
        if (device_index < 0)
          device_index = 0;

        ttnn::Device* device = get_tt_backend().get_device(device_index);
        const Shape shape = {static_cast<dim_t>(size)};
        ttnn::Tensor tensor = ttnn::empty(to_tt_shape(shape),
                                          ttnn::DataType::INT8,
                                          ttnn::Layout::ROW_MAJOR,
                                          device);
        const uint64_t addr = tensor.buffer()->address();
        Allocation allocation;
        allocation.tensor = std::move(tensor);
        allocation.size_bytes = size;
        register_allocation(addr, std::move(allocation));
        return reinterpret_cast<void*>(addr);
      }

      void free(void* ptr, int) override {
        if (!ptr)
          return;
        const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
        unregister_allocation(addr);
      }
    };

  }

  template<>
  Allocator& get_allocator<Device::TT>() {
    static tt::TTAllocator allocator;
    return allocator;
  }

}
