CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB INCLUDE_FILES ../../inc/RawIVEditor/*.[Hh])
FILE(GLOB UI_FORMS *.ui)

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("UI Files" FILES ${UI_FORMS})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

SET(MOC_HEADERS
  ../../inc/RawIVEditor/rawiveditordialog.h
)

if(QT3_FOUND)
  QT_WRAP_CPP(RawIVEditor MOC_SOURCES ${MOC_HEADERS})
  QT_WRAP_UI(RawIVEditor UI_H UI_CPP ${UI_FORMS})

  INCLUDE_DIRECTORIES(
    ../../inc
    ${CMAKE_BINARY_DIR}/src/RawIVEditor  #for generated headers
  )

  ADD_EXECUTABLE(RawIVEditor
    ${MOC_SOURCES}
    ${UI_H}
    ${UI_CPP}
    ${SOURCE_FILES}
    ${INCLUDE_FILES}
  )

  TARGET_LINK_LIBRARIES(RawIVEditor
    ${QT_LIBRARIES}
  )
else(QT3_FOUND)
  message("RawIVEditor only works with Qt3! Not creating RawIVEditor target.")
endif(QT3_FOUND)
