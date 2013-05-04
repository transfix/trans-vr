CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

FILE(GLOB SOURCE_FILES *.cpp)
FILE(GLOB INCLUDE_FILES ../../inc/Filters/*.[Hh])

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
)


# it doesn't look like this really uses Qt... hopefully
# this will just build in Qt4...

#if(QT3_FOUND)
  ADD_LIBRARY(Filters STATIC 
    ${SOURCE_FILES}
    ${INCLUDE_FILES}
  )

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

  target_link_libraries(Filters
    ${QT_LIBRARIES}
  )
#else(QT3_FOUND)
#  message("Filters only works with Qt3! Not creating Filters target.")
#endif(QT3_FOUND)
