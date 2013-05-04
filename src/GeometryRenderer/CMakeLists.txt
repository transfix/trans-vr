cmake_minimum_required(VERSION 2.6)

file(GLOB SOURCE_FILES *.cpp)
file(GLOB INCLUDE_FILES ../../inc/GeometryRenderer/*.[Hh])

include_directories(
  ../../inc
)

SetupQt()

set(LIBS CVC cvcraw_geometry QGLViewer glew) 

source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${INCLUDE_FILES})

add_library(GeometryRenderer STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
SetupBoost(GeometryRenderer)

target_link_libraries(GeometryRenderer
  ${LIBS}	
)
