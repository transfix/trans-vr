cmake_minimum_required(VERSION 2.6)

file(GLOB SOURCE_FILES *.cpp)
file(GLOB INCLUDE_FILES 
	../../inc/Tiling/*.[Hh]
	../../inc/Tiling/ct/*.[Hh]
)

source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${INCLUDE_FILES})

include_directories(
  ../../inc
)

add_library(Tiling STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
SetupBoost(Tiling)
SetupCGAL(Tiling)
SetupGSL(Tiling)
target_link_libraries(Tiling
  cvcraw_geometry # for Geometry class
)
