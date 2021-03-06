cmake_minimum_required(VERSION 2.6)

SetupQt()

if(WIN32)
  add_definitions(-DGLEW_STATIC)
endif(WIN32)

if(ENABLE_STEREO_DISPLAY)
  add_definitions(-DENABLE_STEREO_DISPLAY)
endif(ENABLE_STEREO_DISPLAY)

set(ISOCONTOURING_LIB "")
option(VOLUMEROVER2_ISOCONTOURING_WITH_LBIE "Build VolumeRover2 with LBIE isocontouring support (can call FastContouring depending on method selected)" ON)
mark_as_advanced(VOLUMEROVER2_ISOCONTOURING_WITH_LBIE)
if(VOLUMEROVER2_ISOCONTOURING_WITH_LBIE)
  add_definitions(-DISOCONTOURING_WITH_LBIE)
  set(ISOCONTOURING_LIB LBIE)
endif(VOLUMEROVER2_ISOCONTOURING_WITH_LBIE)

option(VOLUMEROVER2_ISOCONTOURING_WITH_FASTCONTOURING "Build VolumeRover2 with direct FastContouring isocontouring support" ON)
mark_as_advanced(VOLUMEROVER2_ISOCONTOURING_WITH_FASTCONTOURING)
if(VOLUMEROVER2_ISOCONTOURING_WITH_FASTCONTOURING)
  add_definitions(-DISOCONTOURING_WITH_FASTCONTOURING)
  set(ISOCONTOURING_LIB FastContouring)
endif(VOLUMEROVER2_ISOCONTOURING_WITH_FASTCONTOURING)

# Enabling both at the same time isn't a problem now.  In fact, it's preferred!
# This is only an issue for using the deprecated VolumeViewerPage
#if(VOLUMEROVER2_ISOCONTOURING_WITH_LBIE AND VOLUMEROVER2_ISOCONTOURING_WITH_FASTCONTOURING)
#  message(SEND_ERROR "VolumeRover2: Cannot choose both LBIE and FastContouring based isocontouring!")
#endif(VOLUMEROVER2_ISOCONTOURING_WITH_LBIE AND VOLUMEROVER2_ISOCONTOURING_WITH_FASTCONTOURING)

option(VOLUMEROVER2_THREADED_ISOCONTOURING "Run interactive isocontouring routines in a separate thread from the main GUI" ON)
mark_as_advanced(VOLUMEROVER2_THREADED_ISOCONTOURING)
if(VOLUMEROVER2_THREADED_ISOCONTOURING)
  add_definitions(-DTHREADED_ISOCONTOURING)
endif(VOLUMEROVER2_THREADED_ISOCONTOURING)

add_definitions(-DVOLUMEROVER_VERSION_STRING=\"${VolumeRover_VERSION}\")

option(VOLUMEROVER2_FLOAT_COLORTABLE "Use float values for uploaded color table for greater precision.  Only works with CG and ARB volume rendering paths for now" OFF)
mark_as_advanced(VOLUMEROVER2_FLOAT_COLORTABLE)
if(VOLUMEROVER2_FLOAT_COLORTABLE)
  add_definitions(-DVOLUMEROVER2_FLOAT_COLORTABLE)
endif(VOLUMEROVER2_FLOAT_COLORTABLE)

#FILE(GLOB SOURCE_FILES *.cpp)
set(SOURCE_FILES
#  VolumeViewerPage.cpp
  main.cpp
  CVCMainWindow.cpp
  Viewers.cpp
  VolumeViewer.cpp
  BoundingBoxModify.cpp
  ImportData.cpp
#  VolumeRoverMain.cpp
  DimensionModify.cpp
  GeometryInterface.cpp
  ContoursInterface.cpp
  VolumeMemoryInterface.cpp
  NewVolumeDialog.cpp
  VolumeInterface.cpp
  RemapVoxels.cpp
  ReconstructionDialog.cpp
  AnisotropicDiffusionDialog.cpp
  BilateralFilterDialog.cpp
  ContourTilerDialog.cpp
  ContrastEnhancementDialog.cpp
  GDTVFilterDialog.cpp
  PocketTunnelDialog.cpp
  CurationDialog.cpp
  TightCoconeDialog.cpp
  SuperSecondaryStructuresDialog.cpp
  SkeletonizationDialog.cpp
  LBIEQualityImprovementDialog.cpp
  MultiTileServerDialog.cpp
  VolumeViewer2.cpp
  Application.cpp
)
#FILE(GLOB INCLUDE_FILES ../../inc/VolumeRover2/*.[Hh])
set(INCLUDE_FILES
  ../../inc/VolumeRover2/DimensionModify.h
  ../../inc/VolumeRover2/VolMagickEventHandler.h
  ../../inc/VolumeRover2/DataWidget.h
  ../../inc/VolumeRover2/VolumeViewerPage.h
  ../../inc/VolumeRover2/NewVolumeDialog.h
#  ../../inc/VolumeRover2/VolumeRoverMain.h
  ../../inc/VolumeRover2/ImportData.h
  ../../inc/VolumeRover2/BoundingBoxModify.h
  ../../inc/VolumeRover2/CVCMainWindow.h
  ../../inc/VolumeRover2/RemapVoxels.h
  ../../inc/VolumeRover2/VolumeViewer.h
  ../../inc/VolumeRover2/VolumeMemoryInterface.h
  ../../inc/VolumeRover2/Viewers.h
  ../../inc/VolumeRover2/VolumeInterface.h
  ../../inc/VolumeRover2/VolumeRoverExtension.h
  ../../inc/VolumeRover2/MultiTileServerDialog.h
  ../../inc/VolumeRover2/VolumeViewer2.h
)

# arand: this is out of date, remove Qt3 stuff
#FILE(GLOB UI3_FORMS *.Qt3.ui)
set(UI3_FORMS
  boundingboxmodifybase.Qt3.ui
  editvariablebase.Qt3.ui
  dimensionmodifybase.Qt3.ui
  volumeinterfacebase.Qt3.ui
  volumerovermainbase.Qt3.ui
  addtimestepbase.Qt3.ui
  addvariablebase.Qt3.ui
  importdatabase.Qt3.ui
  remapvoxelsbase.Qt3.ui
)
#FILE(GLOB UI4_FORMS *.Qt4.ui)
set(UI4_FORMS
  HLSSurfaceDialog.Qt4.ui
  AnisotropicDiffusionDialog.Qt4.ui
  RemapVoxels.Qt4.ui
  SecondaryStructureDialog.Qt4.ui
  HistogramDialog.Qt4.ui
  CurationDialog.Qt4.ui
#  VolumeRoverMain.Qt4.ui
  ContrastEnhancementDialog.Qt4.ui
  NewVolumeDialog.Qt4.ui
  ImportData.Qt4.ui
  TightCoconeDialog.Qt4.ui
  SuperSecondaryStructuresDialog.Qt4.ui
  DimensionModify.Qt4.ui
  SegmentVirusMapDialog.Qt4.ui
  UnknownData.Qt4.ui
  BoundingBoxModify.Qt4.ui
  BilateralFilterDialog.Qt4.ui
  ContourTilerDialog.Qt4.ui
  PocketTunnelDialog.Qt4.ui
  LBIE_dialog.Qt4.ui
  LBIE_qualityImprovement.Qt4.ui
  ReconstructionDialog.Qt4.ui
  CVCMainWidget.Qt4.ui
  VolumeInterface.Qt4.ui
  GeometryInterface.Qt4.ui
  ContoursInterface.Qt4.ui
  VolumeMemoryInterface.Qt4.ui
  GDTVFilterDialog.Qt4.ui
  SkeletonizationDialog.Qt4.ui
  VolumeViewerPage.Qt4.ui
  VolumeViewerPageManipulators.Qt4.ui
  AddPropertyDialog.Qt4.ui
  MainOptions.Qt4.ui
  SaveFileDialog.Qt4.ui
  SliceRenderingDialog.Qt4.ui
  CloseDataDialog.Qt4.ui
  SaveImageDialog.Qt4.ui
  ViewGeometryOptionsDialog.Qt4.ui
  SelectCurrentVolumeDialog.Qt4.ui
  MultiTileServerDialog.Qt4.ui
  generateRawvDialog.Qt4.ui
  generateMeshDialog.Qt4.ui
  mslevelsetDialog.Qt4.ui
  mplevelsetDialog.Qt4.ui
)

source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${INCLUDE_FILES})

include_directories(
  ../../inc
  ${CMAKE_BINARY_DIR}/src/VolumeRover2  #for generated headers
)

set(MOC_HEADERS
#  ../../inc/VolumeRover2/VolumeRoverMain.h
  ../../inc/VolumeRover2/RemapVoxels.h
  ../../inc/VolumeRover2/ImportData.h
  ../../inc/VolumeRover2/NewVolumeDialog.h
  ../../inc/VolumeRover2/VolumeInterface.h
  ../../inc/VolumeRover2/GeometryInterface.h
  ../../inc/VolumeRover2/ContoursInterface.h
  ../../inc/VolumeRover2/VolumeMemoryInterface.h
  ../../inc/VolumeRover2/DimensionModify.h
#  ../../inc/VolumeRover2/VolumeViewerPage.h
  ../../inc/VolumeRover2/BoundingBoxModify.h
  ../../inc/VolumeRover2/CVCMainWindow.h
  ../../inc/VolumeRover2/Viewers.h
  ../../inc/VolumeRover2/VolumeViewer.h
  ../../inc/VolumeRover2/ReconstructionDialog.h
  ../../inc/VolumeRover2/AnisotropicDiffusionDialog.h
  ../../inc/VolumeRover2/BilateralFilterDialog.h
  ../../inc/VolumeRover2/ContourTilerDialog.h
  ../../inc/VolumeRover2/ContrastEnhancementDialog.h
  ../../inc/VolumeRover2/GDTVFilterDialog.h
  ../../inc/VolumeRover2/PocketTunnelDialog.h
  ../../inc/VolumeRover2/CurationDialog.h
  ../../inc/VolumeRover2/TightCoconeDialog.h
  ../../inc/VolumeRover2/SuperSecondaryStructuresDialog.h
  ../../inc/VolumeRover2/SkeletonizationDialog.h
  ../../inc/VolumeRover2/LBIEQualityImprovementDialog.h
  ../../inc/VolumeRover2/MultiTileServerDialog.h
  ../../inc/VolumeRover2/VolumeViewer2.h
)

# the core CVC lib that contains CVC::App
set(LIBS CVC)

#
# arand: need to fix the CGAL dependence so that CGAL is included once
#        but whenever a library using CGAL is used
#
# joe - 6/10/2011 - added SetupCGAL to make this easier.


if(BUILD_POCKETTUNNEL_LIB AND NOT DISABLE_CGAL)
  set(LIBS ${LIBS} PocketTunnel)
endif(BUILD_POCKETTUNNEL_LIB AND NOT DISABLE_CGAL)

if(BUILD_SWEETMESH_LIB)
  set(LIBS ${LIBS} SweetMesh)
endif(BUILD_SWEETMESH_LIB)

if(BUILD_SKELETONIZATION_LIB AND NOT DISABLE_CGAL)
  set(LIBS ${LIBS} Skeletonization)
endif(BUILD_SKELETONIZATION_LIB AND NOT DISABLE_CGAL)

if(BUILD_TIGHTCOCONE_LIB AND NOT DISABLE_CGAL)
  set(LIBS ${LIBS} TightCocone)
endif(BUILD_TIGHTCOCONE_LIB AND NOT DISABLE_CGAL)

if(BUILD_SUPERSECONDARYSTRUCTURES_LIB AND NOT DISABLE_CGAL)
  set(LIBS ${LIBS} SuperSecondaryStructures)
endif(BUILD_SUPERSECONDARYSTRUCTURES_LIB AND NOT DISABLE_CGAL)

if(BUILD_CURATION_LIB AND NOT DISABLE_CGAL)
  set(LIBS ${LIBS} Curation)
endif(BUILD_CURATION_LIB AND NOT DISABLE_CGAL)

if(BUILD_SECONDARYSTRUCTURES_LIB AND NOT DISABLE_CGAL)
 set(LIBS ${LIBS} SecondaryStructures)
 set(LIBS ${LIBS} Histogram)
endif(BUILD_SECONDARYSTRUCTURES_LIB AND NOT DISABLE_CGAL)

if(BUILD_RECONSTRUCTION_LIB)
  set(LIBS ${LIBS} Reconstruction)
endif(BUILD_RECONSTRUCTION_LIB)
	
if(BUILD_HLEVELSET_LIB)
  set(LIBS ${LIBS} HLevelSet)
endif(BUILD_HLEVELSET_LIB)

if(BUILD_SEGMENTATION_LIB AND NOT DISABLE_CGAL)
  set(LIBS ${LIBS} SegCapsid SegMonomer SegMed SegSubunit SecStruct)
endif(BUILD_SEGMENTATION_LIB AND NOT DISABLE_CGAL)

if(BUILD_TILING_LIB AND NOT DISABLE_CGAL)
  set(LIBS ${LIBS} ContourTiler CGAL CGALcore++ gsl gslcblas mpfr gmp ContourFilterLib)
endif(BUILD_TILING_LIB AND NOT DISABLE_CGAL)

if(BUILD_MMHLS_LIB)
  set(LIBS ${LIBS} ExtMCMesher generateRawV generateMesh)
endif(BUILD_MMHLS_LIB)

if(BUILD_MSLEVELSET_LIB)
  FIND_PACKAGE(CUDA)
  INCLUDE_DIRECTORIES(
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_CUT_INCLUDE_DIR}
  )
  set(LIBS ${LIBS}  ${CUDA_LIBRARIES} ${CUDA_CUT_LIBRARIES} MSLevelSet)
  if(BUILD_HOSEGMENTATION_LIB)
  set(LIBS ${LIBS} HOSegmentation)
  	if(BUILD_MPSEGMENTATION_LIB)
	set(LIBS ${LIBS} MPSegmentation)
	endif(BUILD_MPSEGMENTATION_LIB)
  endif(BUILD_HOSEGMENTATION_LIB)
endif(BUILD_MSLEVELSET_LIB)



# VolumeRover2 dependencies
set(LIBS
    ${LIBS}
    ColorTable2
    ${ISOCONTOURING_LIB}
 # VolumeViewer is deprecated as of 2/13/2011.  New VolumeViewer development should be done on the version
 # in the VolumeRover2 directory for now.  The reason for this is that it is taking advantage of the CVC::App
 # functionality now, and that is only available in VolumeRover2.  In the future we should create a new
 # library for core functionality such as CVC::App.  We should keep linking to VolumeViewer library for as long
 # as we need to keep building the old VolumeViewerPage for reference.
#
# arand, 4-25-2011: it looks like we don't need this anymore
#
#    VolumeViewer
    VolMagick
    cvcraw_geometry
    VolumeRenderer
    GeometryRenderer
    QGLViewer
    cvcalgo
    XmlRPC
#    Socket
    log4cplus
)

#link against VolumeGridRover if it is available
if(BUILD_VOLUMEGRIDROVER AND NOT DISABLE_CGAL) # works in Qt4 now
  include_directories(${CMAKE_BINARY_DIR}/src/VolumeGridRover) # for generated headers
  set(LIBS ${LIBS} VolumeGridRover)
endif(BUILD_VOLUMEGRIDROVER AND NOT DISABLE_CGAL)

if(QT3_FOUND)
  set(UI_FORMS ${UI3_FORMS})
  source_group("UI Files" FILES ${UI_FORMS})

  qt_wrap_cpp(VolumeRover2 MOC_SOURCES ${MOC_HEADERS})
  qt_wrap_ui(VolumeRover2 UI_H UI_CPP ${UI_FORMS})

  add_executable(VolumeRover2 MACOSX_BUNDLE
    ${SOURCE_FILES}
    ${INCLUDE_FILES}
    ${MOC_SOURCES}
    ${UI_H}
    ${UI_CPP}
  )

  target_link_libraries(VolumeRover2 ${LIBS})
elseif(QT4_FOUND)
  set(UI_FORMS ${UI4_FORMS})
  source_group("UI Files" FILES ${UI_FORMS})

  qt4_wrap_cpp(MOC_SOURCES ${MOC_HEADERS})
  qt4_wrap_ui(UI_FILES ${UI_FORMS})

  set(MACOSX_BUNDLE_ICON_FILE ${VolumeRover_SOURCE_DIR}/dat/VolRover.icns)

  add_executable(VolumeRover2 MACOSX_BUNDLE
    ${SOURCE_FILES}
    ${INCLUDE_FILES}
    ${MOC_SOURCES}
    ${UI_FILES}
  )

  target_link_libraries(VolumeRover2 ${LIBS})
endif(QT3_FOUND)

if(BUILD_VOLUMEGRIDROVER AND NOT DISABLE_CGAL) # works in Qt4 now
SetupGSL(VolumeRover2)
endif(BUILD_VOLUMEGRIDROVER AND NOT DISABLE_CGAL) # works in Qt4 now

SetupBoost(VolumeRover2)

if (NOT DISABLE_CGAL)
  SetupCGAL(VolumeRover2)
endif (NOT DISABLE_CGAL)

if(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.4)
  if(APPLE)
    set_target_properties(VolumeRover2 PROPERTIES
      OUTPUT_NAME ${CMAKE_BUNDLE_NAME})
  endif(APPLE)
  set(CMAKE_INSTALL_DESTINATION_ARGS 
    BUNDLE DESTINATION "${CMAKE_BUNDLE_LOCATION}")
endif(${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.4)

install(TARGETS VolumeRover2
        RUNTIME DESTINATION bin ${CMAKE_INSTALL_DESTINATION_ARGS}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

#if(UNIX AND NOT APPLE)
#  install(PROGRAMS #"${VolumeRover_SOURCE_DIR}/bin/runVolumeRover" DESTINATION bin)
#endif(UNIX AND NOT APPLE)

# install rules for including 3rd party libs such as Qt
# if a system Qt is used (e.g. installed in /usr/lib/), it will not be included in the installation
set(fixup_exe "\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin/VolumeRover2${CMAKE_EXECUTABLE_SUFFIX}")

#set(fixup_exe "${CMAKE_INSTALL_PREFIX}/bin/VolumeRover2${CMAKE_EXECUTABLE_SUFFIX}")


if(APPLE)
  set(fixup_exe "\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/MacOS/${CMAKE_BUNDLE_NAME}")
endif(APPLE)

#install(CODE "
#  include(\"${VolumeRover_SOURCE_DIR}/CMake/BundleUtilities.cmake\")
#  fixup_bundle(\"${fixup_exe}\" \"\" \"/usr/local/lib;/Library/Frameworks/Cg.framework;${QT_LIBRARY_DIR};${QT_BINARY_DIR};${Boost_LIBRARY_DIRS};${BUNDLE_LIB_SEARCH_DIRS}\")
#")

#install(CODE "
#  include(\"${VolumeRover_SOURCE_DIR}/CMake/BundleUtilities.cmake\")
#  fixup_bundle(\"${fixup_exe}\" \"\" \"/usr/local/lib;/org/centers/cvc/software/share/usr.linux.x86_64/cg;${QT_LIBRARY_DIR};${QT_BINARY_DIR};${Boost_LIBRARY_DIRS#};${BUNDLE_LIB_SEARCH_DIRS};/org/centers/cvc/software/share/usr.linux.x86_64/cg/lib64/\")
#")

# 8-23-2011, arand: trying to make the CG component of fixup_bundle a little cleaner over the two platforms
find_package(Cg)

install(CODE "
  include(\"${VolumeRover_SOURCE_DIR}/CMake/BundleUtilities.cmake\")
  fixup_bundle(\"${fixup_exe}\" \"\" \"${CG_LIBRARY_DIR};${QT_LIBRARY_DIR};${QT_BINARY_DIR};${Boost_LIBRARY_DIRS};${BUNDLE_LIB_SEARCH_DIRS};/org/centers/cvc/software/share/usr.linux.x86_64/cg/lib64/\")
")

install(CODE "
  include(\"${VolumeRover_SOURCE_DIR}/CMake/BundleUtilities.cmake\")
  fixup_bundle(\"${fixup_exe}\" \"\" \"${CG_LIBRARY_DIR};${QT_LIBRARY_DIR};${QT_BINARY_DIR};${Boost_LIBRARY_DIRS};${BUNDLE_LIB_SEARCH_DIRS}\")
")
