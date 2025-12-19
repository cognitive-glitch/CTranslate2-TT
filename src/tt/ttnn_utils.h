#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <ttnn/ttnn.hpp>

#include "ctranslate2/storage_view.h"
#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace tt {

    inline ttnn::DataType to_tt_dtype(DataType dtype) {
      switch (dtype) {
      case DataType::FLOAT32:
        return ttnn::DataType::FLOAT32;
      case DataType::FLOAT16:
        return ttnn::DataType::FLOAT16;
      case DataType::BFLOAT16:
        return ttnn::DataType::BFLOAT16;
      case DataType::INT8:
        return ttnn::DataType::INT8;
      case DataType::INT16:
        return ttnn::DataType::INT16;
      case DataType::INT32:
        return ttnn::DataType::INT32;
      default:
        throw std::invalid_argument("Unsupported dtype for TT backend");
      }
    }

    inline ttnn::Shape to_tt_shape(const Shape& shape) {
      std::vector<int64_t> dims;
      dims.reserve(shape.size());
      for (const dim_t dim : shape)
        dims.push_back(static_cast<int64_t>(dim));
      return ttnn::Shape(dims);
    }

    inline bool is_tile_compatible(const Shape& shape) {
      for (const dim_t dim : shape) {
        if (dim % 32 != 0)
          return false;
      }
      return !shape.empty();
    }

    inline size_t dtype_size(DataType dtype) {
      switch (dtype) {
      case DataType::FLOAT32:
        return sizeof(float);
      case DataType::FLOAT16:
        return sizeof(float16_t);
      case DataType::BFLOAT16:
        return sizeof(bfloat16_t);
      case DataType::INT8:
        return sizeof(int8_t);
      case DataType::INT16:
        return sizeof(int16_t);
      case DataType::INT32:
        return sizeof(int32_t);
      default:
        throw std::invalid_argument("Unsupported dtype size for TT backend");
      }
    }

  }
}
