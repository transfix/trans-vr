CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB INCLUDE_FILES ../../inc/ColorTable2D/*.[Hh])

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

SET(MOC_HEADERS
  ../../inc/ColorTable2D/TableCanvas.h
  ../../inc/ColorTable2D/ColorTable2D.h
  ../../inc/ColorTable2D/AlphaCanvas.h
  ../../inc/ColorTable2D/ColorCanvas.h
)

if(QT3_FOUND)
  QT_WRAP_CPP(ColorTable2D MOC_SOURCES ${MOC_HEADERS})

  INCLUDE_DIRECTORIES(
    ../../inc
  )

  ADD_LIBRARY (ColorTable2D STATIC 
    ${SOURCE_FILES}
    ${INCLUDE_FILES}
    ${MOC_SOURCES}
  )
else(QT3_FOUND)
  message("ColorTable2D only works with Qt3! Not creating ColorTable2D target.")
endif(QT3_FOUND)
