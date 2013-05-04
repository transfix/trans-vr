CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

#FILE(GLOB SOURCE_FILES *.cpp)
SET(SOURCE_FILES
  Animation.cpp
  AnimationNode.cpp
  ViewState.cpp
) 
SET(APP_SOURCE_FILES
  Animation.cpp
  AnimationNode.cpp
  AnimationWidget.cpp
  GeometryRenderer.cpp
  main.cpp
  MouseSliderHandler.cpp
  RawIVTestRenderable.cpp
  RoverRenderable.cpp
  ViewState.cpp
)
FILE(GLOB INCLUDE_FILES ../../inc/AnimationMaker/*.[Hh])

SOURCE_GROUP("Source Files" FILES ${APP_SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
)

ADD_LIBRARY (AnimationMaker STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
ADD_DEFINITIONS  (${QT_DEFINITIONS})

if(QT3_FOUND)
#  ADD_LIBRARY (AnimationMaker STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
#  ADD_DEFINITIONS  (${QT_DEFINITIONS})
  
  OPTION(BUILD_ANIMATIONMAKER_STANDALONE "Build animation maker standalone app" OFF)
  IF(BUILD_ANIMATIONMAKER_STANDALONE)
    ADD_EXECUTABLE (animation
      ${APP_SOURCE_FILES}
      ${INCLUDE_FILES}
    )

    SetupBoost(animation)

    TARGET_LINK_LIBRARIES(animation
      GeometryFileTypes
      ByteOrder
      VolumeLibrary
      VolumeFileTypes
      VolumeWidget
      Contouring
      Contour
      c2c_codec
      contourtree
      LBIE
      VolMagick
      glew
      ${QT_LIBRARIES}
    )
  ENDIF(BUILD_ANIMATIONMAKER_STANDALONE)
else(QT3_FOUND)
  message("AnimationMaker only works with Qt3! Not create AnimationMaker target (or it's standalone).")
endif(QT3_FOUND)
