CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB INCLUDE_FILES ../../inc/VolumeWidget/*.[Hh])

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
)

SET(MOC_HEADERS
  ../../inc/VolumeWidget/SimpleOpenGLWidget.h
  ../../inc/VolumeWidget/ViewInteractor.h
)



if(QT4_FOUND)

  # arand: need to fix to be Qt4 compatible?
  SET(UI_FORMS ${UI4_FORMS})
  SOURCE_GROUP("UI Files" FILES ${UI_FORMS})
  
  QT4_WRAP_CPP(MOC_SOURCES ${MOC_HEADERS})
  QT4_WRAP_UI(UI_FILES ${UI_FORMS})  

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

  ADD_LIBRARY (VolumeWidget STATIC ${SOURCE_FILES} ${MOC_SOURCES} ${INCLUDE_FILES})

  TARGET_LINK_LIBRARIES(VolumeWidget
    ${QT_LIBRARIES}
    cvcraw_geometry
    LBIE
    glew
  )
endif(QT4_FOUND)


if(QT3_FOUND)
  QT_WRAP_CPP(VolumeWidget MOC_SOURCES ${MOC_HEADERS})

  ADD_LIBRARY (VolumeWidget STATIC ${SOURCE_FILES} ${MOC_SOURCES} ${INCLUDE_FILES})

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

  TARGET_LINK_LIBRARIES(VolumeWidget
    ${QT_LIBRARIES}
    cvcraw_geometry
    LBIE
    glew
  )
endif(QT3_FOUND)
