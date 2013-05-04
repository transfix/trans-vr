CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB INCLUDE_FILES ../../inc/VolumeViewer/*.[Hh])

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
)

SET(MOC_HEADERS
  ../../inc/VolumeViewer/VolumeViewer.h
)

if(QT3_FOUND)
  QT_WRAP_CPP(VolumeViewer MOC_SOURCES ${MOC_HEADERS})
elseif(QT4_FOUND)
  QT4_WRAP_CPP(MOC_SOURCES ${MOC_HEADERS})
endif(QT3_FOUND)

ADD_LIBRARY (VolumeViewer STATIC ${SOURCE_FILES} ${MOC_SOURCES} ${INCLUDE_FILES})

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

TARGET_LINK_LIBRARIES(VolumeViewer
  VolMagick
  cvcraw_geometry
  VolumeLibrary
  QGLViewer
)
