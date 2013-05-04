CMAKE_MINIMUM_REQUIRED (VERSION 2.6.2)

# FindCUDA
# Set CUDA_BUILD_EMULATION to OFF to enable Device mode compilation.
FIND_PACKAGE(CUDA)

FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB CUDA_SOURCE_FILES *.cu)
FILE(GLOB INCLUDE_FILES ../../inc/MSLevelSet/*.[Hh])
FILE(GLOB PRIVATE_INCLUDE_FILES ../../inc/MSLevelSet/private/*.[Hh])
SET(INCLUDE_FILES ${PRIVATE_INCLUDE_FILES} ${INCLUDE_FILES})

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES} ${CUDA_SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

# The include_directories adds paths to both C/C++ compilation in the
# native compiler and cuda compilation in NVCC.
INCLUDE_DIRECTORIES(
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_CUT_INCLUDE_DIR}
#  ${CUDA_SDK_DIR}/common/inc
  ../../inc
)

set(BUILD_SHARED_LIBS OFF)
CUDA_ADD_LIBRARY(MSLevelSet
  ${CUDA_SOURCE_FILES}
  ${SOURCE_FILES}
  ${INCLUDE_FILES}
  # SHARED
  # STATIC
  OPTIONS "-DTEST"
  RELEASE --use_fast_math -DNDEBUG
  DEBUG -g -DDEBUG
)

TARGET_LINK_LIBRARIES(MSLevelSet
 ${CUDA_LIBRARIES}
)

# Add a special target to clean nvcc generated files.
#CUDA_BUILD_CLEAN_TARGET()
