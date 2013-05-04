cmake_minimum_required(VERSION 2.6)

file(GLOB SOURCE_FILES *.cpp)
file(GLOB INCLUDE_FILES ../../inc/SecondaryStructures/*.[Hh])

source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${HEADER_FILES})

include_directories(
  ../../inc
)
add_definitions(-DHUGE=1000000)

find_package(Lapack)
if(LAPACK_FOUND)
  add_library(SecondaryStructures STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
  SetupCGAL(SecondaryStructures)
  target_link_libraries(SecondaryStructures ${LAPACK_LIBRARIES})
else(LAPACK_FOUND)
  message(SEND_ERROR "SecondaryStructures requires Lapack!")
endif(LAPACK_FOUND)
