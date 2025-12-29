#
# This macro is for setting up a sub-project to use the Gnu Scientific Library (GSL)
#

macro(SetupGSL TargetName)
  find_package(GSL)
  if(GSL_FOUND)
    target_include_directories(${TargetName} PRIVATE ${GSL_INCLUDE_DIRS})
    target_compile_options(${TargetName} PRIVATE ${CMAKE_GSL_CXX_FLAGS})
    target_link_libraries(${TargetName} PRIVATE ${GSL_LIBRARIES})
  else(GSL_FOUND)
    message(SEND_ERROR "${TargetName} requires the Gnu Scientific Library (GSL)!")
  endif(GSL_FOUND)
endmacro(SetupGSL)
