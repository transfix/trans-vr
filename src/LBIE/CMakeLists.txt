CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

#FILE(GLOB SOURCE_FILES *.cpp)
SET(SOURCE_FILES
  LBIE_Mesher.cpp
  octree.cpp
  hexa.cpp
  e_face.cpp
  LBIE_geoframe.cpp
  normalspline.cpp
  pcio.cpp
  quality_improve.cpp
  tetra.cpp
)
FILE(GLOB INCLUDE_FILES ../../inc/LBIE/*.[Hh])

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
)

ADD_LIBRARY (LBIE STATIC ${SOURCE_FILES} ${INCLUDE_FILES})

SET(LINK_LIBS
  FastContouring
  Contour
  VolMagick
)

OPTION(USING_LBIE_GEOFRAME_SDF_REFINEMENT "Experimental SDF mesh refinement." OFF)
MARK_AS_ADVANCED(USING_LBIE_GEOFRAME_SDF_REFINEMENT)
IF(USING_LBIE_GEOFRAME_SDF_REFINEMENT)
  ADD_DEFINITIONS(-DUSING_LBIE_GEOFRAME_SDF_REFINEMENT)
  SET(LINK_LIBS ${LINK_LIBS} GeometryFileTypes multi_sdf)
ENDIF(USING_LBIE_GEOFRAME_SDF_REFINEMENT)

#Make sure we've got boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.34.0)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  message("Boost includes: ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message("If you're having trouble finding boost, set environment variables "
          "BOOST_INCLUDEDIR and BOOST_LIBRARYDIR to the appropriate paths")
endif(Boost_FOUND)

TARGET_LINK_LIBRARIES(LBIE ${LINK_LIBS})

OPTION(BUILD_LBIE_MESHER "Build LBIE Mesher" ON)
IF(BUILD_LBIE_MESHER)
  find_package(Boost 1.34.0 COMPONENTS program_options)
  if(Boost_FOUND)
    ADD_EXECUTABLE(Mesher main.cpp)
    TARGET_LINK_LIBRARIES(Mesher
      LBIE
      ${Boost_LIBRARIES} 
    )
  else(Boost_FOUND)
    message("boost_program_options component not found, cannot build Mesher")
  endif(Boost_FOUND)
ENDIF(BUILD_LBIE_MESHER)
