CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt()

if(WIN32)
  add_definitions(-DWIN32)
endif(WIN32)

#assume we're using GCC if not using Visual C++ compiler
if(NOT MSVC)
  add_definitions(-DUSING_GCC)
endif(NOT MSVC)

SET(SOURCE_FILES
  BasicVolumeFileImpl.cpp
  DataCutterSource.cpp
  DownLoadManager.cpp
  MrcFileImpl.cpp
  PifFileImpl.cpp
  RawIVFileImpl.cpp
  RawIVSimpleSource.cpp
  RawVFileImpl.cpp
  SourceManager.cpp
  VolumeBuffer.cpp
  VolumeBufferManager.cpp
  VolumeFile.cpp
  VolumeFileFactory.cpp
  VolumeFileSource.cpp
  VolumeFileSink.cpp
  VolumeTranscriber.cpp
  VolumeSource.cpp
  VolumeSink.cpp
)

SET(INCLUDE_FILES
  ../../inc/VolumeFileTypes/BasicVolumeFileImpl.h
  ../../inc/VolumeFileTypes/DataCutterSource.h
  ../../inc/VolumeFileTypes/DownLoadManager.h
  ../../inc/VolumeFileTypes/MrcFileImpl.h
  ../../inc/VolumeFileTypes/PifFileImpl.h
  ../../inc/VolumeFileTypes/RawIVFileImpl.h
  ../../inc/VolumeFileTypes/RawVFileImpl.h
  ../../inc/VolumeFileTypes/SourceManager.h
  ../../inc/VolumeFileTypes/VolumeBuffer.h
  ../../inc/VolumeFileTypes/VolumeBufferManager.h
  ../../inc/VolumeFileTypes/VolumeFileFactory.h
  ../../inc/VolumeFileTypes/VolumeFile.h
  ../../inc/VolumeFileTypes/VolumeFileSource.h
  ../../inc/VolumeFileTypes/VolumeFileSink.h
  ../../inc/VolumeFileTypes/VolumeTranscriber.h
  ../../inc/VolumeFileTypes/VolumeSource.h
)

if(UNIX)
  add_definitions(-DLARGEFILE_KLUDGE)
  SET(SOURCE_FILES ${SOURCE_FILES} pfile.cpp)
  SET(INCLUDE_FILES ${INCLUDE_FILES} ../../inc/VolumeFileTypes/pfile.h)
endif(UNIX)

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

INCLUDE_DIRECTORIES(
  ../../inc
)

#for libcontour?
ADD_DEFINITIONS(-DARITH_ENCODE -DARITH_DECODE -DUNIQUE -DZP_CODEC)

#if(QT3_FOUND)
  ADD_LIBRARY(VolumeFileTypes STATIC
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

  TARGET_LINK_LIBRARIES(VolumeFileTypes
    contourtree
    dataCutterClient
    Filters
  )
#else(QT3_FOUND)
#  message("VolumeFileTypes only works with Qt3! Not creating VolumeFileTypes target.")
#endif(QT3_FOUND)

