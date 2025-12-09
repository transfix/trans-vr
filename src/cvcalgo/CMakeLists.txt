cmake_minimum_required(VERSION 3.16)

# Setup Qt6 or Qt4
if(USE_QT6)
  find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_AUTOUIC ON)
  set(QT_LIBRARIES Qt6::Core Qt6::Gui Qt6::Widgets)
else()
  SetupQt()
endif()

FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB INCLUDE_FILES ../../inc/cvcalgo/*.[Hh])
FILE(GLOB UI_SOURCE_FILES ui/*.cpp)
FILE(GLOB UI_INCLUDE_FILES ../../inc/cvcalgo/ui/*.[Hh])
FILE(GLOB UI_FORMS ui/*.ui)

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})
SOURCE_GROUP("UI Files" FILES ${UI_FORMS})

INCLUDE_DIRECTORIES(
  ../../inc
  ${CMAKE_BINARY_DIR}/src/cvcalgo  #for generated headers
)

set(MOC_HEADERS
  ../../inc/cvcalgo/ui/SignedDistanceFunctionDialog.h
)

#disable multi_sdf for now until we can link with CGAL on Windows
add_definitions(-DNO_MULTI_SDF)

# cvcalgo dependencies
set(LIBS
  #multi_sdf              # MULTI_SDF
  SignDistanceFunction   # SDFLIB
)

set(FINAL_SOURCES
  ${SOURCE_FILES}
  ${INCLUDE_FILES}
)

# Qt6 uses AUTOMOC/AUTOUIC, Qt4 uses manual wrapping
if(USE_QT6 OR QT4_FOUND)
  set(FINAL_SOURCES
    ${FINAL_SOURCES}
    ${UI_SOURCE_FILES}
    ${UI_INCLUDE_FILES}
    ${UI_FORMS}
  )

  set(LIBS
    ${QT_LIBRARIES}
    ${LIBS}
  )
else()
  message("cvcalgo: Qt not found, not building UI")
endif()

add_library (cvcalgo STATIC ${FINAL_SOURCES})

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

target_link_libraries(cvcalgo ${LIBS})

