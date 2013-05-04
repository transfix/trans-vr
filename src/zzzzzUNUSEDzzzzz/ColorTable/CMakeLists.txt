CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB INCLUDE_FILES ../../inc/ColorTable/*.[Hh])

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
)

SET(MOC_HEADERS
  ../../inc/ColorTable/XoomedOut.h
  ../../inc/ColorTable/XoomedIn.h
  ../../inc/ColorTable/ColorTable.h
)

#if(QT3_FOUND)
#  QT_WRAP_CPP(ColorTable MOC_SOURCES ${MOC_HEADERS})
#
#  ADD_LIBRARY (ColorTable STATIC 
#    ${SOURCE_FILES}
#    ${INCLUDE_FILES}
#    ${MOC_SOURCES}
#  )
#  target_link_libraries(ColorTable ${QT_LIBRARIES})
#else(QT3_FOUND)
#  message("ColorTable only works with Qt3! Not creating ColorTable target.")
#endif(QT3_FOUND)

QT4_WRAP_CPP(MOC_SOURCES ${MOC_HEADERS})

ADD_LIBRARY (ColorTable STATIC 
  ${SOURCE_FILES}
  ${INCLUDE_FILES}
  ${MOC_SOURCES}
)