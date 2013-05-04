CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB INCLUDE_FILES ../../inc/Histogram/*.[Hh])
SET(MOC_HEADERS
  ../../inc/Histogram/histogram.h
  ../../inc/Histogram/glcontrolwidget.h
)

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${HEADER_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
)

QT_WRAP_CPP (Histogram MOC_SOURCES ${MOC_HEADERS})

ADD_LIBRARY (Histogram STATIC 
  ${SOURCE_FILES}
  ${INCLUDE_FILES}
  ${MOC_SOURCES}
)
