#
# This macro is for setting up a sub-project to use CGAL
#

macro(SetupCGAL TargetName)
  if(NOT DISABLE_CGAL)
    find_package(CGAL REQUIRED COMPONENTS Core)
    if(CGAL_FOUND)
      # Modern CGAL 5.x+ uses imported targets, not CGAL_USE_FILE
      # include(${CGAL_USE_FILE})  # Deprecated - causes C++11 features to be disabled
      
      # need the following flags in case CGAL has some special compiler needs for this compiler
      # Note: CGAL 5.x sets these automatically via its imported target
      # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CGAL_CXX_FLAGS_INIT}")
      # set(LIBS ${LIBS} ${CGAL_LIBRARIES})
      
      add_definitions(-DUSING_CGAL)
      if(CMAKE_COMPILER_IS_GNUCXX)
        message("SetupCGAL: g++ detected, using -frounding-math")
        add_definitions(-frounding-math)
      endif(CMAKE_COMPILER_IS_GNUCXX)
      
      # Use modern CGAL imported targets
      target_link_libraries(${TargetName} PUBLIC CGAL::CGAL)
    else(CGAL_FOUND)
      message("${TargetName} is requesting CGAL but it isnt found on the system!")
    endif(CGAL_FOUND)
  endif(NOT DISABLE_CGAL)
endmacro(SetupCGAL)
