#
# Modern Qt6 setup macro for VolumeRover
# This replaces the old SetupQt.cmake for Qt6 migration
#

macro(SetupQt)
  message(STATUS "Setting up Qt6 for VolumeRover...")
  
  # Enable automatic MOC, UIC, RCC for Qt
  # These must be set in each subdirectory due to CMake policy scopes
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_AUTOUIC ON)
  set(CMAKE_AUTORCC ON)
  
  # Find Qt6 with required components
  find_package(Qt6 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
    Xml
    OpenGL
    OpenGLWidgets
  )
  
  if(Qt6_FOUND)
    message(STATUS "Found Qt6 version: ${Qt6_VERSION}")
    message(STATUS "Qt6 Core: ${Qt6Core_DIR}")
  endif()
  
  # Qt6 automatically sets up include directories via targets
  # No need for manual QT_USE_* variables like in Qt4
  
  # For Windows static builds
  if(WIN32)
    if(Qt6_IS_STATIC)
      add_definitions(-DQT_STATICPLUGIN)
      message(STATUS "Using static Qt6 build")
    endif()
  endif(WIN32)
  
  # Clean namespace definition (same as Qt4 version)
  add_definitions(-DQT_CLEAN_NAMESPACE)
  
  # Disable Qt3 support (no longer exists in Qt6)
  add_definitions(-DQT_NO_QT3SUPPORT)
  
  # Set Qt6 found flag for legacy compatibility in CMake files
  set(QT6_FOUND TRUE)
  set(QT4_FOUND FALSE)
  set(QT3_FOUND FALSE)
  
  message(STATUS "Qt6 setup complete")
endmacro(SetupQt)

# Legacy macro - no longer used
macro(SetupQt3)
  message(FATAL_ERROR "Qt3 is no longer supported. Please use Qt6.")
endmacro(SetupQt3)

# Helper function to link Qt6 to a target
function(link_qt6_to_target target_name)
  target_link_libraries(${target_name}
    PRIVATE
      Qt6::Core
      Qt6::Gui
      Qt6::Widgets
      Qt6::Xml
      Qt6::OpenGL
      Qt6::OpenGLWidgets
  )
endfunction()
