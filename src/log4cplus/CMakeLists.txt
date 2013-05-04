#project (log4cplus CXX C)
cmake_minimum_required (VERSION 2.6)
#set (CMAKE_VERBOSE_MAKEFILE on)

set (log4cplus_version_major 1)
set (log4cplus_version_minor 0)
set (log4cplus_version_patch 4)
set (log4cplus_soversion 4)
set (log4cplus_postfix "")

find_package (Threads)
#message (STATUS "Threads: ${CMAKE_THREAD_LIBS_INIT}")

set (log4cplus_headers
  ../../inc/log4cplus/appender.h
  ../../inc/log4cplus/config/macosx.h
  ../../inc/log4cplus/config/win32.h
  ../../inc/log4cplus/config.hxx
  ../../inc/log4cplus/configurator.h
  ../../inc/log4cplus/consoleappender.h
  ../../inc/log4cplus/fileappender.h
  ../../inc/log4cplus/fstreams.h
  ../../inc/log4cplus/helpers/appenderattachableimpl.h
  ../../inc/log4cplus/helpers/loglog.h
  ../../inc/log4cplus/helpers/logloguser.h
  ../../inc/log4cplus/helpers/pointer.h
  ../../inc/log4cplus/helpers/property.h
  ../../inc/log4cplus/helpers/sleep.h
  ../../inc/log4cplus/helpers/socket.h
  ../../inc/log4cplus/helpers/socketbuffer.h
  ../../inc/log4cplus/helpers/stringhelper.h
  ../../inc/log4cplus/helpers/syncprims.h
  ../../inc/log4cplus/helpers/thread-config.h
  ../../inc/log4cplus/helpers/threads.h
  ../../inc/log4cplus/helpers/timehelper.h
  ../../inc/log4cplus/hierarchy.h
  ../../inc/log4cplus/hierarchylocker.h
  ../../inc/log4cplus/layout.h
  ../../inc/log4cplus/logger.h
  ../../inc/log4cplus/loggingmacros.h
  ../../inc/log4cplus/loglevel.h
  ../../inc/log4cplus/ndc.h
  ../../inc/log4cplus/nteventlogappender.h
  ../../inc/log4cplus/nullappender.h
  ../../inc/log4cplus/socketappender.h
  ../../inc/log4cplus/spi/appenderattachable.h
  ../../inc/log4cplus/spi/factory.h
  ../../inc/log4cplus/spi/filter.h
  ../../inc/log4cplus/spi/loggerfactory.h
  ../../inc/log4cplus/spi/loggerimpl.h
  ../../inc/log4cplus/spi/loggingevent.h
  ../../inc/log4cplus/spi/objectregistry.h
  ../../inc/log4cplus/spi/rootlogger.h
  ../../inc/log4cplus/streams.h
  ../../inc/log4cplus/syslogappender.h
  ../../inc/log4cplus/tstring.h
  ../../inc/log4cplus/version.h
  ../../inc/log4cplus/win32debugappender.h
  ../../inc/log4cplus/win32consoleappender.h)


set (log4cplus_sources
  appender.cxx
  appenderattachableimpl.cxx
  configurator.cxx
  consoleappender.cxx
  factory.cxx
  fileappender.cxx
  filter.cxx
  global-init.cxx
  hierarchy.cxx
  hierarchylocker.cxx
  layout.cxx
  logger.cxx
  loggerimpl.cxx
  loggingevent.cxx
  loglevel.cxx
  loglog.cxx
  logloguser.cxx
  ndc.cxx
  nullappender.cxx
  objectregistry.cxx
  patternlayout.cxx
  pointer.cxx
  property.cxx
  rootlogger.cxx
  sleep.cxx
  socket.cxx
  socketappender.cxx
  socketbuffer.cxx
  stringhelper.cxx
  syncprims.cxx
  syslogappender.cxx
  threads.cxx
  timehelper.cxx
  version.cxx)

#message (STATUS "Type: ${UNIX}|${CYGWIN}|${WIN32}")

# Uncomment the following line to get locale aware string conversions.
#add_definitions (-DLOG4CPLUS_WORKING_LOCALE)

if ("${UNIX}" OR "${CYGWIN}")
  set (log4cplus_sources ${log4cplus_sources}
    socket-unix.cxx)
  set (log4cplus_headers ${log4cplus_headers}
    ../../inc/log4cplus/config/defines.hxx
    ../../inc/log4cplus/helpers/syncprims-pthreads.h)
elseif (WIN32)
  set (log4cplus_sources ${log4cplus_sources}
    socket-win32.cxx
    win32debugappender.cxx
    nteventlogappender.cxx
    win32consoleappender.cxx)
  set (log4cplus_headers ${log4cplus_headers}
    ../../inc/log4cplus/helpers/syncprims-win32.h)

  # Comment out the following lines if you want non-Unicode version.
  add_definitions (-DUNICODE -D_UNICODE -UMBCS -U_MBCS)
  set (log4cplus_postfix "${log4cplus_postfix}U")
  
  #add_definitions (-DLOG4CPLUS_STATIC)
  #set (log4cplus_postfix "${log4cplus_postfix}S")
endif ()

#message (STATUS "Headers: ${log4cplus_headers}")
#message (STATUS "Sources: ${log4cplus_sources}")

include_directories (
  ../../inc
)
#include_directories ("include")
set (log4cplus_all_sources ${log4cplus_sources} ${log4cplus_headers})

add_library (log4cplus STATIC ${log4cplus_all_sources})
#add_library (log4cplus SHARED ${log4cplus_all_sources})
target_link_libraries (log4cplus ${CMAKE_THREAD_LIBS_INIT})

set_target_properties (log4cplus PROPERTIES
  VERSION "${log4cplus_version_major}.${log4cplus_version_minor}"
  SOVERSION "${log4cplus_soversion}"
  COMPILE_FLAGS "-DINSIDE_LOG4CPLUS")
if (WIN32)
  set_target_properties (log4cplus PROPERTIES
    DEBUG_POSTFIX "${log4cplus_postfix}D"
    RELEASE_POSTFIX "${log4cplus_postfix}"
    MINSIZEREL_POSTFIX "${log4cplus_postfix}"
    RELWITHDEBINFO_POSTFIX "${log4cplus_postfix}")
  target_link_libraries (log4cplus ws2_32 advapi32)
endif ()

#add_subdirectory (loggingserver)
#add_subdirectory (tests)
