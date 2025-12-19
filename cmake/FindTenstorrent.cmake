# Find the Tenstorrent TTNN/TT-Metal libraries.
#
# The following variables are optionally searched for defaults:
#  TT_METAL_HOME: Base directory where TT-Metalium is installed
#
# The following are set after configuration is done:
#  TENSTORRENT_FOUND
#  TENSTORRENT_INCLUDE_DIRS
#  TENSTORRENT_LIBRARIES

set(_TT_METAL_HOME "$ENV{TT_METAL_HOME}")

find_path(TENSTORRENT_TTNN_INCLUDE_DIR
  NAMES ttnn.hpp
  PATHS
    ${_TT_METAL_HOME}/ttnn/cpp/ttnn
    ${_TT_METAL_HOME}/ttnn
)

find_path(TENSTORRENT_TTMETAL_INCLUDE_DIR
  NAMES tt_metal.hpp
  PATHS
    ${_TT_METAL_HOME}/tt_metal
    ${_TT_METAL_HOME}/tt_metal/include
)

find_library(TENSTORRENT_TTNN_LIBRARY
  NAMES ttnn
  PATHS
    ${_TT_METAL_HOME}/lib
    ${_TT_METAL_HOME}/build/lib
)

find_library(TENSTORRENT_TTMETAL_LIBRARY
  NAMES tt_metal
  PATHS
    ${_TT_METAL_HOME}/lib
    ${_TT_METAL_HOME}/build/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Tenstorrent DEFAULT_MSG
  TENSTORRENT_TTNN_INCLUDE_DIR
  TENSTORRENT_TTNN_LIBRARY
  TENSTORRENT_TTMETAL_LIBRARY
)

if (TENSTORRENT_FOUND)
  set(TENSTORRENT_INCLUDE_DIRS
    ${TENSTORRENT_TTNN_INCLUDE_DIR}
    ${TENSTORRENT_TTMETAL_INCLUDE_DIR}
  )
  set(TENSTORRENT_LIBRARIES
    ${TENSTORRENT_TTNN_LIBRARY}
    ${TENSTORRENT_TTMETAL_LIBRARY}
  )
  message(STATUS "Found Tenstorrent (include: ${TENSTORRENT_INCLUDE_DIRS}, library: ${TENSTORRENT_LIBRARIES})")
  mark_as_advanced(TENSTORRENT_TTNN_INCLUDE_DIR
                   TENSTORRENT_TTMETAL_INCLUDE_DIR
                   TENSTORRENT_TTNN_LIBRARY
                   TENSTORRENT_TTMETAL_LIBRARY)
endif()
