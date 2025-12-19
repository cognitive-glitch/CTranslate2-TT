#include "tt/backend.h"

#include <stdexcept>

namespace ctranslate2 {
  namespace tt {

    TTContext::TTContext() {
      const int num_devices = get_num_devices();
      if (num_devices > 0)
        _devices.resize(num_devices, nullptr);
    }

    TTContext::~TTContext() {
      for (ttnn::Device* device : _devices) {
        if (device)
          ttnn::close_device(device);
      }
    }

    int TTContext::get_num_devices() const {
      return static_cast<int>(ttnn::get_num_devices());
    }

    ttnn::Device* TTContext::get_device(int device_id) {
      if (device_id < 0 || device_id >= get_num_devices())
        throw std::invalid_argument("Invalid TT device index: " + std::to_string(device_id));

      std::lock_guard<std::mutex> lock(_mutex);
      if (_devices.empty())
        _devices.resize(get_num_devices(), nullptr);
      if (_devices[device_id] == nullptr)
        _devices[device_id] = ttnn::open_device(device_id);
      return _devices[device_id];
    }

    void TTContext::synchronize(int device_id) {
      ttnn::Device* device = get_device(device_id);
      ttnn::synchronize_device(device);
    }

    TTContext& get_tt_backend() {
      static TTContext backend;
      return backend;
    }

  }
}
