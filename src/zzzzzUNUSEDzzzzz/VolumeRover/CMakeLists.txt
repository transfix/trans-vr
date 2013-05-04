CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

SetupQt(3)

#Need this for usage with some of Samrat's libraries (TightCocone, PocketTunnel, Skeletonization)
#Should remove this requirement at some point!
ADD_DEFINITIONS(-DHUGE=1000000)

IF(BUILD_VOLUMEGRIDROVER)
  MESSAGE("Building VolumeRover with VolumeGridRover")
  ADD_DEFINITIONS(-DVOLUMEGRIDROVER)
ENDIF(BUILD_VOLUMEGRIDROVER)

OPTION(VOLUMEROVER_DETACHED_VOLUMEGRIDROVER "Build VolumeRover 1.x with the VolumeGridRover window detached from the main window" ON)
mark_as_advanced(VOLUMEROVER_DETACHED_VOLUMEGRIDROVER)
IF(VOLUMEROVER_DETACHED_VOLUMEGRIDROVER)
  ADD_DEFINITIONS(-DDETACHED_VOLUMEGRIDROVER)
ENDIF(VOLUMEROVER_DETACHED_VOLUMEGRIDROVER)

option(VOLUMEROVER_USING_GEOMETRY_LOADER "Have VolumeRover use the geometry loader instead of the currently problematic cvcraw_geometry reader (fails to read *.line files)" ON)
mark_as_advanced(VOLUMEROVER_USING_GEOMETRY_LOADER)
if(VOLUMEROVER_USING_GEOMETRY_LOADER)
  add_definitions(-DUSING_GEOMETRY_LOADER)
endif(VOLUMEROVER_USING_GEOMETRY_LOADER)


#  slicerenderingdialog.Qt3.cpp
SET(SOURCE_FILES
  Axis3DHandler.cpp filelistdialogimpl.cpp GeometryRenderer.cpp GeometryInteractor.cpp
  imagesavedialogimpl.cpp Image.cpp ImageViewer.cpp main-new.cpp MainWindow.cpp
  optionsdialogimpl.cpp RecentFiles.cpp Rover3DWidget.cpp RoverRenderable.cpp
  serverselectordialogimpl.cpp ZoomedInVolume.cpp ZoomedOutVolume.cpp segmentationdialogimpl.cpp
  terminal.cpp SliceRenderable.cpp BoundaryPointCloudDialog.cpp SignedDistanceFunctionDialog.cpp
  SkeletonizationDialog.cpp TightCoconeDialog.cpp CurationDialog.cpp SmoothingDialog.cpp
  LBIEMeshingDialog.cpp LBIEQualityImprovementDialog.cpp projectgeometrydialog.cpp
  anisotropicdiffusiondialog.Qt3.cpp             
  imageviewerbase.Qt3.cpp                   
  serverselectordialog.Qt3.cpp
  bilateralfilterdialog.Qt3.cpp                  
  lbiemeshingdialogbase.Qt3.cpp             
  signeddistancefunctiondialogbase.Qt3.cpp
  boundarypointclouddialogbase.Qt3.cpp           
  lbiequalityimprovementdialogbase.Qt3.cpp  
  skeletonizationdialogbase.Qt3.cpp
  contrastenhancementdialog.Qt3.cpp              
  convertisosurfacetogeometrydialogbase.Qt3.cpp  
#  newvolumemainwindowbase.Qt3.cpp           
  smoothingdialogbase.Qt3.cpp
  curationdialogbase.Qt3.cpp                     
  optionsdialog.Qt3.cpp                     
  terminalbase.Qt3.cpp
  filelistdialog.Qt3.cpp                         
  pedetectiondialog.Qt3.cpp                 
  tightcoconedialogbase.Qt3.cpp
  gdtvfilterdialog.Qt3.cpp                       
  projectgeometrydialogbase.Qt3.cpp         
  volumesavedialog.Qt3.cpp
  highlevelsetrecondialog.Qt3.cpp
  imagesavedialog.Qt3.cpp                        
  segmentationdialog.Qt3.cpp
  VolumeViewerPage.cpp
)

#FILE(GLOB SOURCE_FILES 
#  ./*.cpp
#)

#SET(INCLUDE_FILES
#  ../../inc/VolumeRover/Axis3DHandler.h
#  ../../inc/VolumeRover/filelistdialogimpl.h
#  ../../inc/VolumeRover/imagesavedialogimpl.h
#  ../../inc/VolumeRover/Image.h
#  ../../inc/VolumeRover/ImageViewer.h
#  ../../inc/VolumeRover/newvolumemainwindow.h
#  ../../inc/VolumeRover/optionsdialogimpl.h
#  ../../inc/VolumeRover/GeometryRenderer.h
#  ../../inc/VolumeRover/GeometryInteractor.h
#  ../../inc/VolumeRover/RecentFiles.h
#  ../../inc/VolumeRover/Rover3DWidget.h
#  ../../inc/VolumeRover/RoverRenderable.h
#  ../../inc/VolumeRover/serverselectordialogimpl.h
#  ../../inc/VolumeRover/ZoomedInVolume.h
#  ../../inc/VolumeRover/ZoomedOutVolume.h
#  ../../inc/VolumeRover/segmentationdialogimpl.h
#  ../../inc/VolumeRover/terminal.h
#  ../../inc/VolumeRover/SliceRenderable.h
#  ../../inc/VolumeRover/BoundaryPointCloudDialog.h
#  ../../inc/VolumeRover/SignedDistanceFunctionDialog.h
#  ../../inc/VolumeRover/SkeletonizationDialog.h
#  ../../inc/VolumeRover/TightCoconeDialog.h
#  ../../inc/VolumeRover/CurationDialog.h
#  ../../inc/VolumeRover/SmoothingDialog.h
#  ../../inc/VolumeRover/LBIEMeshingDialog.h
#  ../../inc/VolumeRover/LBIEQualityImprovementDialog.h
#  ../../inc/VolumeRover/projectgeometrydialog.h
#)

FILE(GLOB INCLUDE_FILES 
  ../../inc/VolumeRover/*.[Hh]
)


#SET(UI_FORMS
#  filelistdialog.ui imageviewerbase.ui newvolumemainwindowbase.ui optionsdialog.ui
#  serverselectordialog.ui segmentationdialog.ui imagesavedialog.ui volumesavedialog.ui
#  terminalbase.ui bilateralfilterdialog.ui contrastenhancementdialog.ui pedetectiondialog.ui
#  anisotropicdiffusiondialog.ui slicerenderingdialog.ui highlevelsetrecondialog.ui boundarypointclouddialogbase.ui
#  signeddistancefunctiondialogbase.ui skeletonizationdialogbase.ui tightcoconedialogbase.ui
#  curationdialogbase.ui convertisosurfacetogeometrydialogbase.ui smoothingdialogbase.ui lbiemeshingdialogbase.ui
#  lbiequalityimprovementdialogbase.ui projectgeometrydialogbase.ui gdtvfilterdialog.ui
#)


#SET(MOC_HEADERS
#  ../../inc/VolumeRover/TightCoconeDialog.h
#  ../../inc/VolumeRover/serverselectordialogimpl.h
#  ../../inc/VolumeRover/Rover3DWidget.h
#  ../../inc/VolumeRover/SignedDistanceFunctionDialog.h
#  ../../inc/VolumeRover/terminal.h
#  ../../inc/VolumeRover/ImageViewer.h
#  ../../inc/VolumeRover/projectgeometrydialog.h
#  ../../inc/VolumeRover/SmoothingDialog.h
#  ../../inc/VolumeRover/GeometryInteractor.h
#  ../../inc/VolumeRover/segmentationdialogimpl.h
#  ../../inc/VolumeRover/optionsdialogimpl.h
#  ../../inc/VolumeRover/imagesavedialogimpl.h
#  ../../inc/VolumeRover/newvolumemainwindow.h
#  ../../inc/VolumeRover/LBIEQualityImprovementDialog.h
#  ../../inc/VolumeRover/BoundaryPointCloudDialog.h
#  ../../inc/VolumeRover/filelistdialogimpl.h
#  ../../inc/VolumeRover/SkeletonizationDialog.h
#  ../../inc/VolumeRover/curationdialogbase.Qt3.h
#  ../../inc/VolumeRover/LBIEMeshingDialog.h
#)

SET(MOC_HEADERS
  ../../inc/VolumeRover/tightcoconedialogbase.Qt3.h
  ../../inc/VolumeRover/TightCoconeDialog.h
  ../../inc/VolumeRover/serverselectordialog.Qt3.h
  ../../inc/VolumeRover/serverselectordialogimpl.h
  ../../inc/VolumeRover/Rover3DWidget.h
  ../../inc/VolumeRover/signeddistancefunctiondialogbase.Qt3.h
  ../../inc/VolumeRover/SignedDistanceFunctionDialog.h
  ../../inc/VolumeRover/terminalbase.Qt3.h
  ../../inc/VolumeRover/terminal.h
  ../../inc/VolumeRover/imageviewerbase.Qt3.h
  ../../inc/VolumeRover/ImageViewer.h
  ../../inc/VolumeRover/projectgeometrydialogbase.Qt3.h
  ../../inc/VolumeRover/projectgeometrydialog.h
  ../../inc/VolumeRover/smoothingdialogbase.Qt3.h
  ../../inc/VolumeRover/SmoothingDialog.h
  ../../inc/VolumeRover/GeometryInteractor.h
  ../../inc/VolumeRover/segmentationdialog.Qt3.h
  ../../inc/VolumeRover/segmentationdialogimpl.h
  ../../inc/VolumeRover/optionsdialog.Qt3.h
  ../../inc/VolumeRover/optionsdialogimpl.h
  ../../inc/VolumeRover/imagesavedialog.Qt3.h
  ../../inc/VolumeRover/imagesavedialogimpl.h
#  ../../inc/VolumeRover/newvolumemainwindowbase.Qt3.h
#  ../../inc/VolumeRover/newvolumemainwindow.h
  ../../inc/VolumeRover/MainWindow.h
  ../../inc/VolumeRover/lbiequalityimprovementdialogbase.Qt3.h
  ../../inc/VolumeRover/LBIEQualityImprovementDialog.h
  ../../inc/VolumeRover/boundarypointclouddialogbase.Qt3.h
  ../../inc/VolumeRover/BoundaryPointCloudDialog.h
  ../../inc/VolumeRover/filelistdialog.Qt3.h
  ../../inc/VolumeRover/filelistdialogimpl.h
  ../../inc/VolumeRover/SkeletonizationDialog.h
  ../../inc/VolumeRover/skeletonizationdialogbase.Qt3.h
  ../../inc/VolumeRover/curationdialogbase.Qt3.h
  ../../inc/VolumeRover/CurationDialog.h
  ../../inc/VolumeRover/lbiemeshingdialogbase.Qt3.h
  ../../inc/VolumeRover/LBIEMeshingDialog.h
  ../../inc/VolumeRover/gdtvfilterdialog.Qt3.h
  ../../inc/VolumeRover/highlevelsetrecondialog.Qt3.h
  ../../inc/VolumeRover/volumesavedialog.Qt3.h
  ../../inc/VolumeRover/pedetectiondialog.Qt3.h
  ../../inc/VolumeRover/convertisosurfacetogeometrydialogbase.Qt3.h
  ../../inc/VolumeRover/contrastenhancementdialog.Qt3.h 
  ../../inc/VolumeRover/bilateralfilterdialog.Qt3.h 
  ../../inc/VolumeRover/anisotropicdiffusiondialog.Qt3.h
  ../../inc/VolumeRover2/VolumeViewerPage.h
)


SET(LINK_LIBS
#  ColorTable
  ColorTable2
#  ColorTable2D
  VolMagick
  VolumeFileTypes
  GeometryFileTypes
  VolumeWidget
  VolumeLibrary
  LBIE
#  VolumeGridRover
  GenSeg
  SegCapsid
  SegMonomer
  SegSubunit
  SegMed
  SecStruct
 # AnimationMaker
  Contouring
  multi_sdf
  SignDistanceFunction
  cvcalgo
  RenderServers
  XmlRPC
  VolumeViewer
)


IF(BUILD_PEDETECTION_LIB)
  SET(LINK_LIBS ${LINK_LIBS} PEDetection)
ENDIF(BUILD_PEDETECTION_LIB)

IF(BUILD_POCKETTUNNEL_LIB)
  SET(LINK_LIBS ${LINK_LIBS} PocketTunnel)
ENDIF(BUILD_POCKETTUNNEL_LIB)

#IF(BUILD_TILING_LIB)
#  SET(LINK_LIBS ${LINK_LIBS} Tiling)
#ENDIF(BUILD_TILING_LIB)

IF(BUILD_TIGHTCOCONE_LIB)
  SET(LINK_LIBS ${LINK_LIBS} TightCocone)
ENDIF(BUILD_TIGHTCOCONE_LIB)

IF(BUILD_CURATION_LIB)
  SET(LINK_LIBS ${LINK_LIBS} Curation)
ENDIF(BUILD_CURATION_LIB)

IF(BUILD_HLEVELSET_LIB)
  SET(LINK_LIBS ${LINK_LIBS} HLevelSet)
ENDIF(BUILD_HLEVELSET_LIB)

IF(BUILD_SKELETONIZATION_LIB)
  SET(LINK_LIBS ${LINK_LIBS} Skeletonization)
  SET(SOURCE_FILES ${SOURCE_FILES} SkeletonRenderable.cpp)
  SET(INCLUDE_FILES ${INCLUDE_FILES} ../../inc/VolumeRover/SkeletonRenderable.h)
ENDIF(BUILD_SKELETONIZATION_LIB)

IF(BUILD_SECONDARYSTRUCTURES_LIB)
	SET(LINK_LIBS ${LINK_LIBS} SecondaryStructures Histogram SecondaryStructureDataManager)
	SET(SOURCE_FILES ${SOURCE_FILES})
	SET(INCLUDE_FILES ${INCLUDE_FILES})
ENDIF(BUILD_SECONDARYSTRUCTURES_LIB)

IF(BUILD_MSLEVELSET_LIB)
  SET(UI_FORMS ${UI_FORMS} mslevelsetdialog.ui)
  SET(LINK_LIBS ${LINK_LIBS} MSLevelSet)
ENDIF(BUILD_MSLEVELSET_LIB)

if(BUILD_RECONSTRUCTION_LIB)
  set(UI_FORMS ${UI_FORMS} reconstructiondialog.ui)
  set(LINK_LIBS ${LINK_LIBS} Reconstruction)
  set(SOURCE_FILES ${SOURCE_FILES} reconstructiondialogimpl.cpp)
  set(INCLUDE_FILES ${INCLUDE_FILES} ../../inc/VolumeRover/reconstructiondialogimpl.h)
  set(MOC_HEADERS ${MOC_HEADERS} ../../inc/VolumeRover/reconstructiondialogimpl.h)
endif(BUILD_RECONSTRUCTION_LIB)

#set up CGAL for project_verts.h
find_package(CGAL)
if(CGAL_FOUND)
  include(${CGAL_USE_FILE})
  # need the following flags in case CGAL has some special compiler needs for this compiler
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CGAL_CXX_FLAGS_INIT}")
  set(LINK_LIBS ${LINK_LIBS} ${CGAL_LIBRARIES})
else(CGAL_FOUND)
  message(SEND_ERROR "VolumeRover requires CGAL!")
endif(CGAL_FOUND)

SOURCE_GROUP("Source Files" FILES ${SOURCE_FILES})
SOURCE_GROUP("UI Files" FILES ${UI_FORMS})
SOURCE_GROUP("Header Files" FILES ${INCLUDE_FILES})

  INCLUDE_DIRECTORIES(
    ../../inc
    ${CMAKE_BINARY_DIR}/src/VolumeRover  #for generated headers
  )

  find_package(HDF5 COMPONENTS C CXX)
  if(HDF5_FOUND)
    add_definitions(-DVOLMAGICK_USING_HDF5 ${HDF5_DEFINITIONS})
 #   set(INCLUDE_FILES
  #     ${INCLUDE_FILES}
   #    ../../inc/VolMagick/HDF5_IO.h
 #   )
 #   set(SOURCE_FILES
 #      ${SOURCE_FILES}
 #      HDF5_IO.cpp
 #   )

    include_directories(${HDF5_CXX_INCLUDE_DIR}
		q${HDF5_INCLUDE_DIR})

    SET(VOLMAGICK_LINK_LIBS ${VOLMAGICK_LINK_LIBS} ${HDF5_hdf5_LIBRARY} ${HDF5_hdf5_cpp_LIBRARYq} ${HDF5_LIBRARIES})
  else(HDF5_FOUND)
    message("HDF5 not found! Disabling CVC file support")
  endif(HDF5_FOUND)




#if(QT3_FOUND)
#  QT_WRAP_CPP(Volume MOC_SOURCES ${MOC_HEADERS})
#  QT_WRAP_UI(Volume UI_H UI_CPP ${UI_FORMS})

#  INCLUDE_DIRECTORIES(
#    ../../inc
#    ${CMAKE_BINARY_DIR}/src/VolumeRover  #for generated headers
#    ${CMAKE_BINARY_DIR}/src/VolumeGridRover
#  )
#
#  ADD_EXECUTABLE (VolumeRover
#    ${MOC_SOURCES}
#    ${UI_H}
#    ${UI_CPP}
#    ${SOURCE_FILES}
#    ${INCLUDE_FILES}
#  )
#
  #Make sure we've got boost
#  set(Boost_USE_STATIC_LIBS OFF)
#  set(Boost_USE_MULTITHREADED ON)
#  find_package(Boost 1.34.0)
#  if(Boost_FOUND)
#    include_directories(${Boost_INCLUDE_DIRS})
#    message("Boost includes: ${Boost_INCLUDE_DIRS}")
#  else(Boost_FOUND)
#    message("If you're having trouble finding boost, set environment variables "
#            "BOOST_INCLUDEDIR and BOOST_LIBRARYDIR to the appropriate paths")
#  endif(Boost_FOUND)
#
#  TARGET_LINK_LIBRARIES(VolumeRover
#    ${LINK_LIBS}
#    ${QT_LIBRARIES}
#  )
#endif(QT3_FOUND)

if(QT4_FOUND)
  FILE(GLOB UI4_FORMS ./*.Qt4.ui)

  INCLUDE_DIRECTORIES(
    ../../inc
    ${CMAKE_BINARY_DIR}/src/VolumeRover  #for generated headers
    ${CMAKE_BINARY_DIR}/src/VolumeGridRover
  )

  SET(UI_FORMS ${UI4_FORMS})
  SOURCE_GROUP("UI Files" FILES ${UI_FORMS})
  
  QT4_WRAP_CPP(MOC_SOURCES ${MOC_HEADERS})
  QT4_WRAP_UI(UI_FILES ${UI_FORMS})  

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

ADD_EXECUTABLE (VolumeRover
  ${SOURCE_FILES}
  ${INCLUDE_FILES}
  ${MOC_SOURCES}
  ${MOC_HEADERS}
  ${UI_H}
  ${UI_CPP}
  ${UI_FILES}
) 

 TARGET_LINK_LIBRARIES(VolumeRover
    ${LINK_LIBS}
    ${QT_LIBRARIES}
  )
endif(QT4_FOUND)
