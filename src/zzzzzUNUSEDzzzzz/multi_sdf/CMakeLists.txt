CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

#FILE(GLOB SOURCE_FILES *.cpp)
set(SOURCE_FILES
  dt.cpp
  init.cpp
  tcocone.cpp
  robust_cc.cpp
  kdtree.cpp
  mesh_io.cpp
  op.cpp
  sdf.cpp
  util.cpp
  rcocone.cpp
  priorityqueue.cpp
  multi_sdf.cpp  
  matrix.cpp
)
FILE(GLOB INCLUDE_FILES ../../inc/multi_sdf/*.[Hh])

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
)

ADD_LIBRARY (multi_sdf STATIC ${SOURCE_FILES} ${INCLUDE_FILES})

set(LIBS
  cvcraw_geometry
  VolMagick
)

find_package(CGAL)
if(CGAL_FOUND)
  include(${CGAL_USE_FILE})
  # need the following flags in case CGAL has some special compiler needs for this compiler
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CGAL_CXX_FLAGS_INIT}")
  set(LIBS ${LIBS} ${CGAL_LIBRARIES})
else(CGAL_FOUND)
  message(SEND_ERROR "multi_sdf requires CGAL!")
endif(CGAL_FOUND)

target_link_libraries(multi_sdf
  ${LIBS}
)

option(BUILD_MULTISDF_STANDALONE "Build standalone multi_sdf program." OFF)
mark_as_advanced(BUILD_MULTISDF_STANDALONE)
if(BUILD_MULTISDF_STANDALONE)
  source_group("Source Files" FILES ${SOURCE_FILES} main.cpp)
  add_executable(multiSDF main.cpp)
  target_link_libraries(multiSDF multi_sdf)    
endif(BUILD_MULTISDF_STANDALONE)
