cmake_minimum_required(VERSION 2.6)

file(GLOB SOURCE_FILES *.cpp)
file(GLOB INCLUDE_FILES ../../inc/Skeletonization/*.[Hh])

source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${INCLUDE_FILES})

include_directories(
  ../../inc
)

add_definitions(-DHUGE=1000000)

add_library(Skeletonization STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
SetupCGAL(Skeletonization)
