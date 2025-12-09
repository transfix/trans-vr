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

file(GLOB SOURCE_FILES *.cpp)
file(GLOB INCLUDE_FILES ../../inc/RawIVEditor/*.[Hh])
file(GLOB UI_FORMS *.ui)

source_group("Source Files" FILES ${SOURCE_FILES})
source_group("UI Files" FILES ${UI_FORMS})
source_group("Header Files" FILES ${INCLUDE_FILES})

set(MOC_HEADERS
  ../../inc/RawIVEditor/rawiveditordialog.h
)

include_directories(
  ../../inc
  ${CMAKE_BINARY_DIR}/src/RawIVEditor  #for generated headers
)

add_executable(RawIVEditor
  ${SOURCE_FILES}
  ${INCLUDE_FILES}
  ${UI_FORMS}
)

target_link_libraries(RawIVEditor
  ${QT_LIBRARIES}
)
