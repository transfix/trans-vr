cmake_minimum_required(VERSION 3.16)

SetupQt()

file(GLOB SOURCE_FILES *.cpp)
file(GLOB INCLUDE_FILES ../../inc/ColorTable2/*.[Hh])

source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${INCLUDE_FILES})

include_directories(
  ../../inc
)

# AUTOMOC will handle MOC generation automatically for headers with Q_OBJECT
add_library(ColorTable2 STATIC 
  ${SOURCE_FILES}
  ${INCLUDE_FILES}
)

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

set(LIBS QGLViewer)

option(COLORTABLE2_DISABLE_CONTOUR_TREE "ColorTable2 -- enable/disable contour tree.  Requires contourtree and VolMagick libs" OFF)
mark_as_advanced(COLORTABLE2_DISABLE_CONTOUR_TREE)
if(COLORTABLE2_DISABLE_CONTOUR_TREE)
  add_definitions(-DCOLORTABLE2_DISABLE_CONTOUR_TREE)
else()
  set(LIBS ${LIBS} contourtree VolMagick)
endif()

option(COLORTABLE2_DISABLE_CONTOUR_SPECTRUM "ColorTable2 -- enable/disable contour spectrum.  Requires libContour and VolMagick libs" OFF)
mark_as_advanced(COLORTABLE2_DISABLE_CONTOUR_SPECTRUM)
if(COLORTABLE2_DISABLE_CONTOUR_SPECTRUM)
  add_definitions(-DCOLORTABLE2_DISABLE_CONTOUR_SPECTRUM)
else()
  set(LIBS ${LIBS} Contour VolMagick)
endif()

target_link_libraries(ColorTable2 PUBLIC ${LIBS})
