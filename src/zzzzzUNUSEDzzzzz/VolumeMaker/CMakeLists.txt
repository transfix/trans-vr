CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB INCLUDE_FILES ../../inc/VolumeMaker/*.[Hh])
FILE(GLOB UI_FILES *.ui)

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
  ${CMAKE_BINARY_DIR}/src/VolumeMaker  #for generated headers
)

SET(MOC_HEADERS
  ../../inc/VolumeMaker/VolumeMaker.h
  ../../inc/VolumeMaker/RemapVoxels.h
  ../../inc/VolumeMaker/ImportData.h
  ../../inc/VolumeMaker/NewVolumeDialog.h
  ../../inc/VolumeMaker/VolumeInterface.h
  ../../inc/VolumeMaker/DimensionModify.h
  ../../inc/VolumeMaker/BoundingBoxModify.h
)

if(QT3_FOUND)
  QT_WRAP_CPP(VolumeMaker MOC_SOURCES ${MOC_HEADERS})
  QT_WRAP_UI(VolumeMaker UI_H UI_CPP ${UI_FILES})

  ADD_EXECUTABLE (VolumeMaker 
    ${SOURCE_FILES}
    ${INCLUDE_FILES}
    ${MOC_SOURCES}
    ${UI_H}
    ${UI_CPP}
  )

  #Make sure we've got boost
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_MULTITHREADED ON)
  find_package(Boost 1.34.0 COMPONENTS filesystem system)
  if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    message("Boost includes: ${Boost_INCLUDE_DIRS}")
    message("Boost libraries: ${Boost_LIBRARIES}")
  else(Boost_FOUND)
    message("If you're having trouble finding boost, set environment variables "
            "BOOST_INCLUDEDIR and BOOST_LIBRARYDIR to the appropriate paths")
  endif(Boost_FOUND)

  TARGET_LINK_LIBRARIES(VolumeMaker
    VolMagick
    ${Boost_LIBRARIES}
    ${QT_LIBRARIES}
  )
else(QT3_FOUND)
  message("VolumeMaker only works with Qt3! Not creating VolumeMaker target.")
endif(QT3_FOUND)
