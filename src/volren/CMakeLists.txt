CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

#FILE(GLOB SOURCE_FILES *.c)
set(SOURCE_FILES vr_util.c vr_init.c vr_io.c vr_raytrac.c vr_set.c)
FILE(GLOB INCLUDE_FILES ../../inc/volren/*.[Hh])

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

find_package(OpenMP)
if(OPENMP_FOUND)
  add_definitions(-DUSE_OMP)
  set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS})
  set(CMAKE_LD_FLAGS ${CMAKE_LD_FLAGS} -lgomp)
endif(OPENMP_FOUND)

INCLUDE_DIRECTORIES(
  ../../inc
)

ADD_LIBRARY (volren STATIC ${SOURCE_FILES} ${INCLUDE_FILES})

target_link_libraries(volren iso m)

option(BUILD_STANDALONE_VOLUME_RAYCASTER "Build standalone raycaster binary" OFF)
if(BUILD_STANDALONE_VOLUME_RAYCASTER)
  add_executable(renvol main.c)
  target_link_libraries(renvol
    volren
    iso
  )
endif(BUILD_STANDALONE_VOLUME_RAYCASTER)
