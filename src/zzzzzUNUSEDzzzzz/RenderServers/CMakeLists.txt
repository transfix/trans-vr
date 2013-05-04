CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

SET(SOURCE_FILES
  FrameInformation.cpp
  raycastserversettingsdialogimpl.cpp
  RenderServer.cpp
  textureserversettingsdialogimpl.cpp
  TransferArray.cpp
)

SET(INCLUDE_FILES
  ../../inc/RenderServers/FrameInformation.h
  ../../inc/RenderServers/raycastserversettingsdialogimpl.h
  ../../inc/RenderServers/RenderServer.h
  ../../inc/RenderServers/textureserversettingsdialogimpl.h
  ../../inc/RenderServers/TransferArray.h
)

SET(MOC_HEADERS
  ../../inc/RenderServers/textureserversettingsdialogimpl.h
  ../../inc/RenderServers/raycastserversettingsdialogimpl.h
)

#SET(UI_FORMS
#  textureserversettingsdialog.ui
#  raycastserversettingsdialog.ui
#)

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("UI Files" FILES ${UI_FORMS})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

set(UI_FILES "")
#if(QT3_FOUND)
#  QT_WRAP_CPP(RenderServers MOC_SOURCES ${MOC_HEADERS})
#  QT_WRAP_UI(RenderServers UI_H UI_CPP ${UI_FORMS})
#  set(UI_FILES ${UI_H} ${UI_CPP})
#
#  ADD_LIBRARY (RenderServers STATIC 
#    ${MOC_SOURCES}
#    ${UI_FILES}
#    ${SOURCE_FILES}
#    ${INCLUDE_FILES}
#  )
#
#  target_link_libraries(RenderServers ${QT_LIBRARIES})
#elseif(QT4_FOUND)
  qt4_wrap_cpp(MOC_SOURCES ${MOC_HEADERS})
  qt4_wrap_ui(UI_FILES ${UI_FORMS})

  ADD_LIBRARY (RenderServers STATIC 
    ${MOC_SOURCES}
    ${UI_FILES}
    ${SOURCE_FILES}
    ${INCLUDE_FILES}
  )

#endif(QT3_FOUND)

INCLUDE_DIRECTORIES(
  ../../inc
  ${CMAKE_BINARY_DIR}/src/RenderServers  #for generated headers
)

MESSAGE(STATUS "TODO: Add CORBA support")

