CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SET(SOURCE_FILES
  octree.cpp
  meshGeneration.cpp
)

SET(INCLUDE_FILES
  ../../inc/SweetLBIE/octree.h
  ../../inc/SweetLBIE/meshGeneration.h
)

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
)

ADD_LIBRARY(SweetLBIE STATIC ${SOURCE_FILES} ${INCLUDE_FILES})

TARGET_LINK_LIBRARIES(SweetLBIE SweetMesh VolMagick)
