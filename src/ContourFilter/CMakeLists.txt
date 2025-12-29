cmake_minimum_required(VERSION 3.16)

SetupQt()

#FILE(GLOB SOURCE_FILES *.cc)
FILE(GLOB INCLUDE_FILES 
	../../inc/ContourFilter/*.[Hh]
)
#SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})
SET(SRCS 
  container.cc 
  contour.cc 
  controls.cc 
  histogram.cc
  object.cc
  point.cc
#  reconstruct2contourtiler.cc
  control_points.cc
  parameter.cc
  sim_anneal.cc
)

INCLUDE_DIRECTORIES (
  ../../inc/ContourFilter
)

SET(LINK_LIBS 
)

SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wshadow -Wextra -Weffc++ -O3 -g -ffast-math -DNOASSERT=true")

ADD_LIBRARY(ContourFilterLib STATIC ${SRCS})
ADD_EXECUTABLE (ContourFilterBin ${SRCS} reconstruct2contourtiler.cc)
#ADD_EXECUTABLE (ContourTilerTest ${TEST_SRCS} test_main.cpp)
#ADD_EXECUTABLE (ContourTilerSandbox sandbox.cpp)

#TARGET_LINK_LIBRARIES(ContourTiler ContourTilerLib ${LINK_LIBS} ${Boost_LIBRARIES})
#TARGET_LINK_LIBRARIES(ContourTilerTest ContourTilerLib ${LINK_LIBS} ${Boost_LIBRARIES})
#TARGET_LINK_LIBRARIES(ContourTilerSandbox ContourTilerLib ${LINK_LIBS} ${Boost_LIBRARIES})

SET_TARGET_PROPERTIES(ContourFilterLib PROPERTIES OUTPUT_NAME "ContourFilter")
SET_TARGET_PROPERTIES(ContourFilterBin PROPERTIES OUTPUT_NAME "ContourFilter")
