CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

#build static for now (The following define is in config.h)
#add_definitions(-DQGLVIEWER_STATIC)

if(WIN32)
  add_definitions(-DWIN32)
endif(WIN32)

if(ENABLE_STEREO_DISPLAY)
  add_definitions(-DENABLE_STEREO_DISPLAY)
endif(ENABLE_STEREO_DISPLAY)

FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB INCLUDE_FILES ../../inc/QGLViewer/*.[Hh])

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
  ${CMAKE_BINARY_DIR}/src/QGLViewer  #for generated headers
)

SET(MOC_HEADERS
  ../../inc/QGLViewer/qglviewer.h
  ../../inc/QGLViewer/frame.h
  ../../inc/QGLViewer/keyFrameInterpolator.h
  ../../inc/QGLViewer/manipulatedFrame.h
  ../../inc/QGLViewer/manipulatedCameraFrame.h
  ../../inc/QGLViewer/camera.h
)

IF(QT4_FOUND)
  SET(UI_FORMS ImageInterface.Qt4.ui)
ELSEIF(QT3_FOUND)
  SET(UI_FORMS ImageInterface.Qt3.ui)
ENDIF(QT4_FOUND)

OPTION(USING_QGLVIEWER_VECTORIAL_RENDERER "Flag for enabling VRender code for QGLViewer" ON)
MARK_AS_ADVANCED(USING_QGLVIEWER_VECTORIAL_RENDERER)
IF(USING_QGLVIEWER_VECTORIAL_RENDERER)
  FILE(GLOB VRENDER_SOURCE_FILES VRender/*.cpp)
  FILE(GLOB VRENDER_INCLUDE_FILES ../../inc/QGLViewer/VRender/*.h)
  SET(SOURCE_FILES ${SOURCE_FILES} ${VRENDER_SOURCE_FILES})
  SET(INCLUDE_FILES ${INCLUDE_FILES} ${VRENDER_INCLUDE_FILES})
  IF(QT4_FOUND)
    SET(UI_FORMS ${UI_FORMS} VRenderInterface.Qt4.ui)
  ELSEIF(QT3_FOUND)
    SET(UI_FORMS ${UI_FORMS} VRenderInterface.Qt3.ui)
  ENDIF(QT4_FOUND)
ELSE(USING_QGLVIEWER_VECTORIAL_RENDERER)
  ADD_DEFINITIONS(-DNO_VECTORIAL_RENDER)
ENDIF(USING_QGLVIEWER_VECTORIAL_RENDERER)

#Might need glut for some versions of Qt apparently.
find_package(GLUT)

IF(GLUT_FOUND)
  INCLUDE_DIRECTORIES(
    ../../inc
    ${GLUT_INCLUDE_DIR}
  )
ELSE(GLUT_FOUND)
  INCLUDE_DIRECTORIES(
    ../../inc
  )
ENDIF(GLUT_FOUND)

set(UI_FILES "")
if(QT4_FOUND)
  qt4_wrap_cpp(MOC_SOURCES ${MOC_HEADERS})
  qt4_wrap_ui(UI_FILES ${UI_FORMS})
elseif(QT3_FOUND)
  qt_wrap_cpp(QGLViewer MOC_SOURCES ${MOC_HEADERS})
  qt_wrap_ui(QGLViewer UI_H UI_CPP ${UI_FORMS})
  set(UI_FILES ${UI_H} ${UI_CPP})
endif(QT4_FOUND)

ADD_LIBRARY (QGLViewer STATIC 
  ${SOURCE_FILES}
  ${INCLUDE_FILES}
  ${UI_FILES}
  ${MOC_SOURCES}
  ${MOC_HEADERS}
)

IF(GLUT_FOUND)
  TARGET_LINK_LIBRARIES(QGLViewer
    ${QT_LIBRARIES}
    ${GLUT_LIBRARIES}
    glew
  )
ELSE(GLUT_FOUND)
  TARGET_LINK_LIBRARIES(QGLViewer
    ${QT_LIBRARIES}
    glew
  )
ENDIF(GLUT_FOUND)
