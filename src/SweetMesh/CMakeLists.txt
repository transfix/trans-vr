CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQT()

SET(SOURCE_FILES
  vertex.cpp
  triangle.cpp
  tetrahedra.cpp
  quad.cpp
  hexahedron.cpp
  hexmesh.cpp
  meshIO.cpp
  volRoverDisplay.cpp
  meshTools.cpp
  hexmeshtest.cpp
)

SET(INCLUDE_FILES
  ../../inc/SweetMesh/vertex.h
  ../../inc/SweetMesh/triangle.h
  ../../inc/SweetMesh/tetrahedra.h
  ../../inc/SweetMesh/quad.h
  ../../inc/SweetMesh/hexahedron.h
  ../../inc/SweetMesh/hexmesh.h
  ../../inc/SweetMesh/meshIO.h
  ../../inc/SweetMesh/volRoverDisplay.h
  ../../inc/SweetMesh/meshTools.h
  ../../inc/SweetMesh/hexmeshtest.h
)

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
)

ADD_LIBRARY (SweetMesh STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
TARGET_LINK_LIBRARIES(SweetMesh cvcraw_geometry)
#OPTION(BUILD_SWEETMESH_STANDALONE "Build standalone SweetMesh app." OFF)

#IF(NOT BUILD_SWEETMESH_STANDALONE)
#  TARGET_LINK_LIBRARIES(SweetMesh cvcraw_geometry)
#ENDIF(NOT BUILD_SWEETMESH_STANDALONE)

#IF(BUILD_SWEETMESH_STANDALONE)
  #ADD_DEFINITIONS(-DUSING_STANDALONE_SWEETMESH)
#  ADD_EXECUTABLE(SweetMesh_standalone main.cpp)
#  TARGET_LINK_LIBRARIES(SweetMesh_standalone SweetMesh)
#ENDIF(BUILD_SWEETMESH_STANDALONE)

#make sure we have boost
SetupBoost(SweetMesh)


