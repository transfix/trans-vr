CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

FILE(GLOB SOURCE_FILES 
   main.cpp Control.cpp GVF.cpp Initialization.cpp EM.cpp Geometric.cpp
   TFGeneration.cpp Evaluation.cpp Stack.cpp PEDetection.cpp
   MarchingCubes.cpp Skeleton.cpp Thinning.cpp
   FrontPlane.cpp MembraneSeg.cpp Octree.cpp OctreeCell.cpp
   VesselSeg.cpp CriticalPoints.cpp STree.cpp
)
FILE(GLOB INCLUDE_FILES ../../inc/PEDetection/*.[Hh])

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
)

ADD_LIBRARY (PEDetection STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
