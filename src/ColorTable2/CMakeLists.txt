CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB INCLUDE_FILES ../../inc/ColorTable2/*.[Hh])

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
)

SET(MOC_HEADERS
  ../../inc/ColorTable2/XoomedOut.h
  ../../inc/ColorTable2/Table.h
  ../../inc/ColorTable2/ColorTable.h
  ../../inc/ColorTable2/InfoDialog.h
)

if(QT3_FOUND)
  QT_WRAP_CPP(ColorTable2 MOC_SOURCES ${MOC_HEADERS})
else(QT4_FOUND)
  qt4_wrap_cpp(MOC_SOURCES ${MOC_HEADERS})
endif(QT3_FOUND)

ADD_LIBRARY (ColorTable2 STATIC 
  ${SOURCE_FILES}
  ${INCLUDE_FILES}
  ${MOC_SOURCES}
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
else(COLORTABLE2_DISABLE_CONTOUR_TREE)
  set(LIBS ${LIBS} contourtree VolMagick)
endif(COLORTABLE2_DISABLE_CONTOUR_TREE)

option(COLORTABLE2_DISABLE_CONTOUR_SPECTRUM "ColorTable2 -- enable/disable contour spectrum.  Requires libContour and VolMagick libs" OFF)
mark_as_advanced(COLORTABLE2_DISABLE_CONTOUR_SPECTRUM)
if(COLORTABLE2_DISABLE_CONTOUR_SPECTRUM)
  add_definitions(-DCOLORTABLE2_DISABLE_CONTOUR_SPECTRUM)
else(COLORTABLE2_DISABLE_CONTOUR_SPECTRUM)
  set(LIBS ${LIBS} Contour VolMagick)
endif(COLORTABLE2_DISABLE_CONTOUR_SPECTRUM)

TARGET_LINK_LIBRARIES(ColorTable2 ${LIBS})
