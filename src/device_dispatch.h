#pragma once

#include <stdexcept>

#include "ctranslate2/devices.h"

#define UNSUPPORTED_DEVICE_CASE(DEVICE)                       \
  case DEVICE: {                                              \
    throw std::runtime_error("unsupported device " #DEVICE);  \
    break;                                                    \
  }

#define DEVICE_CASE(DEVICE, STMT)               \
  case DEVICE: {                                \
    constexpr Device D = DEVICE;                \
    STMT;                                       \
    break;                                      \
  }

#define SINGLE_ARG(...) __VA_ARGS__
#ifdef CT2_WITH_TENSTORRENT
#  define TT_DEVICE_CASE(DEVICE, STMTS)                 \
    DEVICE_CASE(Device::TT, SINGLE_ARG(STMTS))
#else
#  define TT_DEVICE_CASE(DEVICE, STMTS)                 \
    UNSUPPORTED_DEVICE_CASE(Device::TT)
#endif

#ifndef CT2_WITH_CUDA
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    UNSUPPORTED_DEVICE_CASE(Device::CUDA)               \
  /* Tenstorrent backend. */                            \
  /* Keep TT case close to CPU/CUDA for readability. */ \
  /* Guard with build flag to allow compilation without TT SDK. */ \
  /* NOLINTNEXTLINE(bugprone-macro-parentheses) */       \
    TT_DEVICE_CASE(DEVICE, STMTS)                       \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#else
#  define DEVICE_DISPATCH(DEVICE, STMTS)                \
  switch (DEVICE) {                                     \
    DEVICE_CASE(Device::CUDA, SINGLE_ARG(STMTS))        \
  /* Tenstorrent backend. */                            \
  /* NOLINTNEXTLINE(bugprone-macro-parentheses) */       \
    TT_DEVICE_CASE(DEVICE, STMTS)                       \
    DEVICE_CASE(Device::CPU, SINGLE_ARG(STMTS))         \
  }
#endif
