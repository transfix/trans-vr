# $Id: CMakeLists.txt 1405 2010-02-26 21:16:13Z transfix $

CMAKE_MINIMUM_REQUIRED (VERSION 2.6.2)
PROJECT (VolumeRover)

# Global build variables.  Use the standard CPPFLAGS and LDFLAGS env vars.
option(USING_MAKE_ENV_VARS "If this is enabled, use the CPPFLAGS and LDFLAGS environment variables" OFF)
if(USING_MAKE_ENV_VARS)
  string(COMPARE NOTEQUAL "$ENV{CPPFLAGS}" "" CPPFLAGS)
  string(COMPARE NOTEQUAL "$ENV{LDFLAGS}" "" LDFLAGS)
  if(CPPFLAGS AND LDFLAGS)
    set(CMAKE_CXX_FLAGS "-Wno-deprecated -Wno-non-template-friend")
    set(CMAKE_CXX_FLAGS $ENV{CPPFLAGS})
    set(CMAKE_LD_FLAGS $ENV{LDFLAGS})
  endif(CPPFLAGS AND LDFLAGS)
endif(USING_MAKE_ENV_VARS)
option(USING_MAKE_ENV_VARS_FOR_INSTALL "If this is enabled, use the LDFLAGS environment variable as a directory to look up libs for a bundle" ON)
mark_as_advanced(USING_MAKE_ENV_VARS_FOR_INSTALL)
if(USING_MAKE_ENV_VARS_FOR_INSTALL)
  set(ENV_LIB_DIR "$ENV{ENV}/lib")
endif(USING_MAKE_ENV_VARS_FOR_INSTALL)

# Set the osx development target
IF(APPLE)
 option(USING_MACOSX_DEV_TARGET "If this is enabled, you need to specify the mac osx target for build" OFF)
 IF(USING_MACOSX_DEV_TARGET)
   set(MACOSX_DEPLOYMENT_TARGET "10.5")
   set(CMAKE_OSX_SYSROOT "/Developer/SDKs/MaxOSX10.5.sdk")
   set(CMAKE_CXX_FLAGS "--sysroot ${CMAKE_OSX_SYSROOT} ${CMAKE_CXX_FLAGS}")
 ENDIF(USING_MACOSX_DEV_TARGET)
ENDIF(APPLE)

if(NOT BUNDLE_LIB_SEARCH_DIRS)
  set(BUNDLE_LIB_SEARCH_DIRS "${ENV_LIB_DIR}")
endif(NOT BUNDLE_LIB_SEARCH_DIRS)
message(STATUS "BUNDLE_LIB_SEARCH_DIRS: ${BUNDLE_LIB_SEARCH_DIRS}")

# CMake Modules
SET(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/CMake
  ${CMAKE_SOURCE_DIR}/CMake/cuda
  ${CMAKE_MODULE_PATH})

# Build output 
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
mark_as_advanced(LIBRARY_OUTPUT_PATH EXECUTABLE_OUTPUT_PATH)

# Set the revision number from svn.
option(USING_SUBVERSION "If you're using subversion, VolRover can use the revision number for version info" OFF)
set(CVC_SVN_REVISION "0")
if(USING_SUBVERSION)
  find_package(Subversion)
  if(Subversion_FOUND)
    Subversion_WC_INFO(${CMAKE_SOURCE_DIR} VolumeRover)
    set(CVC_SVN_REVISION ${VolumeRover_WC_REVISION})
    message(STATUS "Building Revision: ${CVC_SVN_REVISION}")
  endif(Subversion_FOUND)
endif(USING_SUBVERSION)

# The VolumeRover version number.
SET(VolumeRover_VERSION_MAJOR 2)
SET(VolumeRover_VERSION_MINOR 0)
SET(VolumeRover_VERSION_PATCH ${CVC_SVN_REVISION})

# The full version string
SET(VolumeRover_VERSION ${VolumeRover_VERSION_MAJOR}.${VolumeRover_VERSION_MINOR}.${VolumeRover_VERSION_PATCH})
message(STATUS "VolumeRover version: ${VolumeRover_VERSION}")

# OSX Bundle settings
IF(APPLE)
  SET(CMAKE_BUNDLE_NAME
    "VolumeRover-${VolumeRover_VERSION_MAJOR}.${VolumeRover_VERSION_MINOR}.${VolumeRover_VERSION_PATCH}")
  SET(CMAKE_BUNDLE_LOCATION "${CMAKE_INSTALL_PREFIX}")
  # make sure CMAKE_INSTALL_PREFIX ends in /
  STRING(LENGTH "${CMAKE_INSTALL_PREFIX}" LEN)
  MATH(EXPR LEN "${LEN} -1" )
  STRING(SUBSTRING "${CMAKE_INSTALL_PREFIX}" ${LEN} 1 ENDCH)
  IF(NOT "${ENDCH}" STREQUAL "/")
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/")
  ENDIF(NOT "${ENDCH}" STREQUAL "/")
  SET(CMAKE_INSTALL_PREFIX 
    "${CMAKE_INSTALL_PREFIX}${CMAKE_BUNDLE_NAME}.app/Contents")
ENDIF(APPLE)

# include macros for setting up some library dependencies for sub-projects of VolRover
include( SetupQt )
include( SetupFFTW )
include( SetupBoost )
include( SetupGSL )

option(DISABLE_CGAL "This disables CGAL even if it is available on the build system." OFF)
if (NOT DISABLE_CGAL)
  include( SetupCGAL )
endif (NOT DISABLE_CGAL)

# 05/11/2012 - transfix
# Possibly prevent crashing on OSX Lion
# http://groups.google.com/group/boost-list/browse_thread/thread/1c5a614a9e8c9808
if(APPLE)
  add_definitions(-D_GLIBCXX_FULLY_DYNAMIC_STRING)
endif(APPLE)

# ************* CPack configuration for VolumeRover *************
# Include the packaging system for distributing binaries
if(WIN32)
  set(CPACK_GENERATOR "NSIS")
elseif(APPLE)
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY 1)
  set(CPACK_GENERATOR "ZIP")
else(WIN32)
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY 1)
  set(CPACK_GENERATOR "TGZ")
endif(WIN32)
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Volumetric image visualization and processing")
SET(CPACK_PACKAGE_VENDOR "Computational Visualization Center")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/README")
SET(CPACK_PACKAGE_VERSION_MAJOR "${VolumeRover_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${VolumeRover_VERSION_MINOR}")
SET(CPACK_PACKAGE_VERSION_PATCH "${VolumeRover_VERSION_PATCH}")
SET(CPACK_PACKAGE_VERSION "${VolumeRover_VERSION}")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "VolumeRover ${VolumeRover_VERSION_MAJOR}.${VolumeRover_VERSION_MINOR}")
SET(CPACK_SOURCE_PACKAGE_FILE_NAME "VolumeRover-${VolumeRover_VERSION}")
SET(CPACK_PACKAGE_NAME "${CMAKE_PROJECT_NAME}")
# Installers for 32- vs. 64-bit CMake:
#  - Root install directory (displayed to end user at installer-run time)
#  - "NSIS package/display name" (text used in the installer GUI)
#  - Registry key used to store info about the installation
IF(CMAKE_CL_64)
  SET(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
  SET(CPACK_NSIS_PACKAGE_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} (Win64)")
  SET(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION} (Win64)")
ELSE()
  SET(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
  SET(CPACK_NSIS_PACKAGE_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY}")
  SET(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} VolumeRover")
  SET(CPACK_NSIS_HELP_LINK "http:\\\\\\\\cvcweb.ices.utexas.edu")
  SET(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\cvcweb.ices.utexas.edu")
  SET(CPACK_NSIS_CONTACT "transfix@ices.utexas.edu")
  SET(CPACK_NSIS_MODIFY_PATH ON)
  SET(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION}")
ENDIF()
IF(NOT DEFINED CPACK_SYSTEM_NAME)
  # make sure package is not Cygwin-unknown, for Cygwin just
  # cygwin is good for the system name
  IF("${CMAKE_SYSTEM_NAME}" STREQUAL "CYGWIN")
    SET(CPACK_SYSTEM_NAME Cygwin)
  ELSE("${CMAKE_SYSTEM_NAME}" STREQUAL "CYGWIN")
    SET(CPACK_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})
  ENDIF("${CMAKE_SYSTEM_NAME}" STREQUAL "CYGWIN")
ENDIF(NOT DEFINED CPACK_SYSTEM_NAME)
IF(${CPACK_SYSTEM_NAME} MATCHES Windows)
  IF(CMAKE_CL_64)
    SET(CPACK_SYSTEM_NAME win64-x64)
  ELSE(CMAKE_CL_64)
    SET(CPACK_SYSTEM_NAME win32-x86)
  ENDIF(CMAKE_CL_64)
ENDIF(${CPACK_SYSTEM_NAME} MATCHES Windows)
IF(NOT DEFINED CPACK_PACKAGE_FILE_NAME)
  # if the CPACK_PACKAGE_FILE_NAME is not defined by the cache
  # default to source package - system, on cygwin system is not 
  # needed
  IF(CYGWIN)
    SET(CPACK_PACKAGE_FILE_NAME "${CPACK_SOURCE_PACKAGE_FILE_NAME}")
  ELSE(CYGWIN)
    SET(CPACK_PACKAGE_FILE_NAME 
      "${CPACK_SOURCE_PACKAGE_FILE_NAME}-${CPACK_SYSTEM_NAME}")
  ENDIF(CYGWIN)
ENDIF(NOT DEFINED CPACK_PACKAGE_FILE_NAME)

SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Jose Rivera") #required
SET(CPACK_PACKAGE_CONTACT "transfix@ices.utexas.edu")
SET(CPACK_STRIP_FILES ON)
SET(CPACK_SOURCE_STRIP_FILES "")
SET(CPACK_PACKAGE_EXECUTABLES "VolumeRover2" "VolumeRover")
SET(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README")
SET(CPACK_RESOURCE_FILE_WELCOME "${CMAKE_SOURCE_DIR}/README")
include( CPack )
include(InstallRequiredSystemLibraries)

# experimental
option(USING_STANDARD_INSTALL_LOCATION "If on, it will not put everything under $prefix/VolumeRover2" ON)
if(USING_STANDARD_INSTALL_LOCATION)
  set(ADDED_PREFIX ".")
else(USING_STANDARD_INSTALL_LOCATION)
  set(ADDED_PREFIX "VolumeRover2")
endif(USING_STANDARD_INSTALL_LOCATION)

ADD_SUBDIRECTORY (src)
