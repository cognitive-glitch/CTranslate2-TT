#pragma once

#include <mutex>
#include <vector>

#include "tt/ttnn_utils.h"

namespace ctranslate2 {
  namespace tt {

    class TTContext {
    public:
      TTContext();
      ~TTContext();

      ttnn::Device* get_device(int device_id);
      int get_num_devices() const;
      void synchronize(int device_id);

    private:
      mutable std::mutex _mutex;
      std::vector<ttnn::Device*> _devices;
    };

    TTContext& get_tt_backend();

  }
}
