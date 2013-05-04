CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

#Ensure large file support
ADD_DEFINITIONS(-D_FILE_OFFSET_BITS=64)

SET(SOURCE_FILES
  VolumeGridRover.cpp
  PointClassFile.cpp
  ContourFile.cpp
  SDF2D.cpp
#  bspline_opt.cpp
#  bspline_fit.cpp
  sdf_opt.cpp
)

SET(INCLUDE_FILES
  ../../inc/VolumeGridRover/VolumeGridRover.h
  ../../inc/VolumeGridRover/SurfRecon.h
  ../../inc/VolumeGridRover/PointClassFile.h
  ../../inc/VolumeGridRover/ContourFile.h
  ../../inc/VolumeGridRover/SDF2D.h
#  ../../inc/VolumeGridRover/bspline_opt.h
  ../../inc/VolumeGridRover/sdf_opt.h
)

FILE(GLOB UI3_FORMS *.Qt3.ui)
FILE(GLOB UI4_FORMS *.Qt4.ui)

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
  ${CMAKE_BINARY_DIR}/src/VolumeGridRover  #for generated headers
  C:/include
)

OPTION(VOLUMEGRIDROVER_USING_MEDAX "Compile VolumeGridRover using the medial axis code" ON)
MARK_AS_ADVANCED(VOLUMEGRIDROVER_USING_MEDAX)
IF(VOLUMEGRIDROVER_USING_MEDAX)
  SET(INCLUDE_FILES
    ${INCLUDE_FILES}
    ../../inc/VolumeGridRover/medax.h
  )
  ADD_DEFINITIONS(-DUSING_VOLUMEGRIDROVER_MEDAX)

  OPTION(VOLUMEGRIDROVER_MEDAX_INSERT_EDGES "Use the 'insert edges' option of the medial axis code" ON)
  mark_as_advanced(VOLUMEGRIDROVER_MEDAX_INSERT_EDGES)
  IF(VOLUMEGRIDROVER_MEDAX_INSERT_EDGES)
    ADD_DEFINITIONS(-DMEDAX_INSERT_EDGES)
  ENDIF(VOLUMEGRIDROVER_MEDAX_INSERT_EDGES)

  OPTION(VOLUMEGRIDROVER_MEDAX_ALWAYS_INSERT "Use the 'always insert' option of the medial axis code" ON)
  mark_as_advanced(VOLUMEGRIDROVER_MEDAX_ALWAYS_INSERT)
  IF(VOLUMEGRIDROVER_MEDAX_ALWAYS_INSERT)
    ADD_DEFINITIONS(-DMEDAX_ALWAYS_INSERT)
  ENDIF(VOLUMEGRIDROVER_MEDAX_ALWAYS_INSERT)
ENDIF(VOLUMEGRIDROVER_USING_MEDAX)

OPTION(VOLUMEGRIDROVER_EM_CLUSTERING "Compile VolumeGridRover using Sangmin's EM clustering - causes some serious memory problems if enabled!!" OFF)
MARK_AS_ADVANCED(VOLUMEGRIDROVER_EM_CLUSTERING)
IF(VOLUMEGRIDROVER_EM_CLUSTERING)
  #Note: EM.cpp is kept around for reference.  It isn't actually used.

  SET(INCLUDE_FILES
    ${INCLUDE_FILES}
    ../../inc/VolumeGridRover/EM.h
  )

  ADD_DEFINITIONS(-DUSING_EM_CLUSTERING)
ENDIF(VOLUMEGRIDROVER_EM_CLUSTERING)

SET(MOC_HEADERS
  ../../inc/VolumeGridRover/VolumeGridRover.h
  ../../inc/VolumeGridRover/bspline_opt.h
  ../../inc/VolumeGridRover/sdf_opt.h
)

if(QT3_FOUND)
  SET(UI_FORMS ${UI3_FORMS})
  SOURCE_GROUP("UI Files" FILES ${UI_FORMS})

  QT_WRAP_CPP(VolumeGridRover MOC_SOURCES ${MOC_HEADERS})
  QT_WRAP_UI(VolumeGridRover UI_H UI_CPP ${UI_FORMS})

  ADD_LIBRARY (VolumeGridRover STATIC 
    ${MOC_SOURCES}
    ${UI_H}
    ${UI_CPP}
    ${SOURCE_FILES}
    ${INCLUDE_FILES}
  )

else(QT4_FOUND)
  SET(UI_FORMS ${UI4_FORMS})
  SOURCE_GROUP("UI Files" FILES ${UI_FORMS})

  qt4_wrap_cpp(MOC_SOURCES ${MOC_HEADERS})
  qt4_wrap_ui(UI_FILES ${UI_FORMS})

  add_library(VolumeGridRover STATIC
    ${SOURCE_FILES}
    ${INCLUDE_FILES}
    ${MOC_SOURCES}
    ${UI_FILES}
  )

endif(QT3_FOUND)

SetupBoost(VolumeGridRover)
SetupCGAL(VolumeGridRover)
SetupGSL(VolumeGridRover)

if(NOT BUILD_SEGMENTATION_LIB)
  message(SEND_ERROR "VolumeGridRover requires the Virus segmentation library!")
endif(NOT BUILD_SEGMENTATION_LIB)

SET(LINK_LIBS
  VolMagick
  ColorTable2
  GenSeg
  XmlRPC
  QGLViewer
)

#If we're using Tiling, the VolumeGridRover needs to link
#with Tiling and GeometryFileTypes.
# 11/3/11 edwardsj - converting to new contour tiler
IF(BUILD_TILING_LIB)
  SET(LINK_LIBS
    ${LINK_LIBS}
    ContourFilterLib
    ContourTiler
    ContourFilterLib
#    Tiling
#    GeometryFileTypes
  )
ENDIF(BUILD_TILING_LIB)

#If we're enabling 2D isocontouring, we need to link with
#libContour
option(VOLUMEGRIDROVER_ISOCONTOURING "Enable 2D isocontouring via the VolumeGridRover" ON)
mark_as_advanced(VOLUMEGRIDROVER_ISOCONTOURING)
if(VOLUMEGRIDROVER_ISOCONTOURING)
  add_definitions(-DVOLUMEGRIDROVER_ISOCONTOURING)  
  set(LINK_LIBS
    ${LINK_LIBS}
    Contour
  )
endif(VOLUMEGRIDROVER_ISOCONTOURING)

target_link_libraries(VolumeGridRover ${LINK_LIBS})

OPTION(BUILD_VOLUMEGRIDROVER_STANDALONE "Build standalone VolumeGridRover app." OFF)
IF(BUILD_VOLUMEGRIDROVER_STANDALONE)
  SET(INCLUDE_FILES
    ../../inc/VolumeGridRover/VolumeGridRoverMainWindow.h
  )
  SET(SOURCE_FILES
    VolumeGridRoverMainWindow.cpp
    main.cpp
  )
  SET(UI_FORMS
    VolumeGridRoverMainWindowBase.ui
  )

  SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
  SOURCE_GROUP("UI Files" FILES ${UI_FORMS})
  SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

  if(QT3_FOUND)
    QT_WRAP_CPP(GridRover MOC_SOURCES ${INCLUDE_FILES})
    QT_WRAP_UI(GridRover UI_H UI_CPP ${UI_FORMS})
    ADD_EXECUTABLE(GridRover 
       ${INCLUDE_FILES}
       ${SOURCE_FILES}
       ${UI_H}
       ${UI_CPP}
       ${MOC_SOURCES}
    )

    TARGET_LINK_LIBRARIES(GridRover VolumeGridRover)
  else(QT3_FOUND)

  QT_WRAP_CPP(GridRover MOC_SOURCES ${MOC_HEADERS})

    ADD_EXECUTABLE(GridRover 
       ${INCLUDE_FILES}
       ${SOURCE_FILES}
       ${UI_H}
       ${UI_CPP}
       ${MOC_SOURCES}
    )


  TARGET_LINK_LIBRARIES(GridRover VolumeGridRover)


  endif(QT3_FOUND)
ENDIF(BUILD_VOLUMEGRIDROVER_STANDALONE)
