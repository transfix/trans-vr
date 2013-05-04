CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt() #this library only needs Qt for the UI

FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB INCLUDE_FILES ../../inc/cvcalgo/*.[Hh])
FILE(GLOB UI_SOURCE_FILES ui/*.cpp)
FILE(GLOB UI_INCLUDE_FILES ../../inc/cvcalgo/ui/*.[Hh])
FILE(GLOB UI3_FORMS ui/*.Qt3.ui)
FILE(GLOB UI4_FORMS ui/*.Qt4.ui)

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

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

if(QT3_FOUND)
  message("cvcalgo: No Qt3 dialogs yet!")

  #set(UI_FORMS ${UI3_FORMS})
  #SOURCE_GROUP("UI Files" FILES ${UI_FORMS})

  #QT_WRAP_CPP(cvcalgo MOC_SOURCES ${MOC_HEADERS})
  #QT_WRAP_UI(cvcalgo UI_H UI_CPP ${UI_FORMS})

  #set(FINAL_SOURCES
  #  ${FINAL_SOURCES}
  #  ${UI_SOURCE_FILES}
  #  ${UI_INCLUDE_FILES}
  #  ${MOC_SOURCES}
  #  ${UI_H}
  #  ${UI_CPP}
  #)

  #set(LIBS
  #  ${QT_LIBRARIES}
  #  ${LIBS}
  #)
elseif(QT4_FOUND)
  set(UI_FORMS ${UI4_FORMS})
  source_group("UI Files" FILES ${UI_FORMS})

  qt4_wrap_cpp(MOC_SOURCES ${MOC_HEADERS})
  qt4_wrap_ui(UI_FILES ${UI_FORMS})

  set(FINAL_SOURCES
    ${FINAL_SOURCES}
    ${UI_SOURCE_FILES}
    ${UI_INCLUDE_FILES}
    ${MOC_SOURCES}
    ${UI_FILES}
  )

  set(LIBS
    ${QT_LIBRARIES}
    ${LIBS}
  )
else(QT3_FOUND)
  message("cvcalgo: Qt not found, not building UI")
endif(QT3_FOUND)

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

