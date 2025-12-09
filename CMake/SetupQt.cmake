#
# This macro is for setting up a sub-project to use Qt4 (Qt3 no longer supported)
#

macro(SetupQt)
  # Find and setup Qt4 for this project
  # Note: For new development, use Qt6 with USE_QT6=ON instead
  find_package(Qt4 COMPONENTS QtCore QtGui QtXml QtOpenGL Qt3Support REQUIRED)
  set(QT_USE_QTXML      TRUE)
  set(QT_USE_QT3SUPPORT TRUE) 
  set(QT_USE_QTOPENGL   TRUE)
  set(QT_USE_QTCORE     TRUE)
  set(QT_USE_QTGUI      TRUE)
  include(${QT_USE_FILE})
  set(QT3_FOUND FALSE)
  set(QT4_FOUND TRUE)
  add_definitions(${QT_DEFINITIONS})
  add_definitions(-DQT_CLEAN_NAMESPACE)
  include_directories(${QT_INCLUDE_DIR})
  include_directories(${QT_QT_INCLUDE_DIR})
  include_directories(${QT_QTCORE_INCLUDE_DIR})
  
  if(WIN32)
    add_definitions(-DQT_NODLL)
  endif(WIN32)
endmacro(SetupQt)

macro(SetupQt3)
  message(FATAL_ERROR "Qt3 is no longer supported. Use Qt4 (SetupQt) or Qt6 (USE_QT6=ON)")
endmacro(SetupQt3)
