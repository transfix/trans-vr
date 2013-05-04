TEMPLATE	= app
LANGUAGE	= C++

CONFIG	+= link_prl qt warn_on opengl thread windows rtti exceptions

win32:CONFIG	+= console

DEFINES	+= VOLUMEGRIDROVER DETACHED_VOLUMEGRIDROVER

INCLUDEPATH	+= ../VolumeWidget ../VolumeLibrary ../RenderServers ../ByteOrder ../ColorTable ../Contouring ../libLBIE ../Filters ../VolumeFileTypes ../GeometryFileTypes ../libcontourtree ../c2c_codec ../arithlib ../libdjvu++ ../AnimationMaker ../ColorTable2D ../VolumeGridRover ../XmlRPC ../dataCutterClient ../Segmentation/SegCapsid ../Segmentation/SegMonomer ../Segmentation/SegSubunit ../Segmentation/SecStruct ../VolMagick ../QGLViewer ../glew ../Tiling ../multi_sdf ../SignDistanceFunction ../HLevelSet ../libcontour ../FastContouring

SOURCES	+= Axis3DHandler.cpp filelistdialogimpl.cpp GeometryRenderer.cpp GeometryInteractor.cpp imagesavedialogimpl.cpp Image.cpp ImageViewer.cpp main.cpp newvolumemainwindow.cpp optionsdialogimpl.cpp RecentFiles.cpp Rover3DWidget.cpp RoverRenderable.cpp serverselectordialogimpl.cpp ZoomedInVolume.cpp ZoomedOutVolume.cpp segmentationdialogimpl.cpp terminal.cpp SliceRenderable.cpp BoundaryPointCloudDialog.cpp SignedDistanceFunctionDialog.cpp SkeletonizationDialog.cpp TightCoconeDialog.cpp CurationDialog.cpp SmoothingDialog.cpp LBIEMeshingDialog.cpp LBIEQualityImprovementDialog.cpp quality_improve.cpp projectgeometrydialog.cpp

HEADERS	+= Axis3DHandler.h filelistdialogimpl.h imagesavedialogimpl.h Image.h ImageViewer.h newvolumemainwindow.h optionsdialogimpl.h GeometryRenderer.h GeometryInteractor.h RecentFiles.h Rover3DWidget.h RoverRenderable.h serverselectordialogimpl.h ZoomedInVolume.h ZoomedOutVolume.h segmentationdialogimpl.h terminal.h SliceRenderable.h BoundaryPointCloudDialog.h SignedDistanceFunctionDialog.h SkeletonizationDialog.h TightCoconeDialog.h CurationDialog.h SmoothingDialog.h LBIEMeshingDialog.h LBIEQualityImprovementDialog.h quality_improve.h projectgeometrydialog.h

FORMS	= filelistdialog.ui \
	imageviewerbase.ui \
	newvolumemainwindowbase.ui \
	optionsdialog.ui \
	serverselectordialog.ui \
	segmentationdialog.ui \
	imagesavedialog.ui \
	volumesavedialog.ui \
	terminalbase.ui \
	bilateralfilterdialog.ui \
	contrastenhancementdialog.ui \
	pedetectiondialog.ui \
	anisotropicdiffusiondialog.ui \
	slicerenderingdialog.ui \
	highlevelsetrecondialog.ui \
	boundarypointclouddialogbase.ui \
	signeddistancefunctiondialogbase.ui \
	skeletonizationdialogbase.ui \
	tightcoconedialogbase.ui \
	curationdialogbase.ui \
	convertisosurfacetogeometrydialogbase.ui \
	smoothingdialogbase.ui \
	lbiemeshingdialogbase.ui \
	lbiequalityimprovementdialogbase.ui \
	projectgeometrydialogbase.ui \
	gdtvfilterdialog.ui

# $Id: Volume.pro 1528 2010-03-12 22:28:08Z transfix $


#add extra code for enabled features
contains( DEFINES, USING_SKELETONIZATION ) {
	SOURCES += SkeletonRenderable.cpp
	HEADERS += SkeletonRenderable.h
}

#not sure what HUGE is for in pocketTunnel... doesnt work on win32-g++ without this definition
win32-g++ {
DEFINES += HUGE=1000000
}

TARGET = VolumeRover

!win32 | win32-g++ {
LIBS = \
	../Contouring/libContouring.a \
	../duallib/libdual.a \
	../libLBIE/libLBIE.a \
	../FastContouring/libFastContouring.a \
	../ColorTable/libColorTable.a \
	../ColorTable2D/libColorTable2D.a \
	../VolumeWidget/libVolumeWidget.a \
	../VolumeLibrary/libVolumeLibrary.a \
	../RenderServers/libRenderServers.a \
	../VolumeFileTypes/libVolumeFileTypes.a \
	../dataCutterClient/libdataCutterClient.a \
	../ByteOrder/libByteOrder.a \
	../Filters/libFilters.a \
	../GeometryFileTypes/libGeometryFileTypes.a \
	../libcontour/libcontour.a \
	../libcontourtree/libcontourtree.a \
	../c2c_codec/libc2c_codec.a \
	../arithlib/libarith.a \
	../libdjvu++/libdjvu++.a \
	../AnimationMaker/libAnimation.a \
	../VolumeGridRover/libVolumeGridRover.a \
	../QGLViewer/libQGLViewer.a \
	../glew/libglew.a \
	../XmlRPC/libXmlRPC.a \
	../Segmentation/GenSeg/libGenSeg.a \
	../Segmentation/SegCapsid/libSegCapsid.a \
	../Segmentation/SegMonomer/libSegMonomer.a \
	../Segmentation/SegSubunit/libSegSubunit.a \
	../Segmentation/SecStruct/libSecStruct.a \
	../multi_sdf/libmulti_sdf.a \
	../SignDistanceFunction/libSignDistanceFunction.a \
	../VolMagick/libVolMagick.a
#win32:LIBS += -lwsock32
#LIBS += -lCGAL -lgsl -lgslcblas -lboost_regex -lboost_filesystem
}

#Sangmins Pulmonary Embolus Detection code
contains( DEFINES, USING_PE_DETECTION ) {
	LIBS += ../PEDetection/libPEDetection.a
	INCLUDEPATH += ../PEDetection
}

#Samrats Pocket Tunnel code
contains( DEFINES, USING_POCKET_TUNNEL ) {
	LIBS += ../PocketTunnel/libPocketTunnel.a #-lCGAL
	INCLUDEPATH += ../PocketTunnel
}

contains( DEFINES, USING_TILING ) {
	LIBS += ../Tiling/libTiling.a
	INCLUDEPATH += ../Tiling
}

contains( DEFINES, USING_TIGHT_COCONE ) {
	LIBS += ../TightCocone/libTightCocone.a
}

contains( DEFINES, USING_CURATION ) {
	LIBS += ../Curation/libCuration.a
}

contains( DEFINES, USING_HLEVELSET ) {
	LIBS += ../HLevelSet/libHLevelSet.a
}

contains( DEFINES, USING_SKELETONIZATION ) {
	LIBS += ../Skeletonization/libSkeletonization.a
}

contains( DEFINES, USING_MSLEVELSET) {
	FORMS += mslevelsetdialog.ui
	LIBS += ../MSLevelSet/libMSLevelSet.a -lcudart -lcutil
	CUDA_DIR = $$(CUDA_DIR)
	isEmpty(CUDA_DIR) {
		# auto-detect CUDA path
		CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
	}
	message( "Ensure that this is the correct CUDA path: $${CUDA_DIR}")
	QMAKE_LIBDIR += $$CUDA_DIR/lib
}

contains( DEFINES, USING_HDF5 ){
	LIBS += -lhdf5_cpp -lhdf5
}

!win32 | win32-g++ {
win32:LIBS += -lwsock32
LIBS += -lCGAL -lgmp -lgsl -lgslcblas -lboost_regex -lboost_filesystem -lboost_system
}

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

#setup CORBA stuff
MYOOCDIR = $$(OOCDIR)
#MYOOCDIR =
isEmpty(MYOOCDIR) {

#check if the define was set...

	contains( DEFINES, USING_CORBA) {
		DEFINES += USINGCORBA
 	}

	contains( DEFINES, USINGCORBA) {
		message("Using system ORBACUS via CPPFLAGS and LDFLAGS")
		message("If you have corba in a location different than what you have specified in CPPFLAGS or LDFLAGS, use the OOCDIR environment variable")
		LIBS += -lCosNaming -lOB
	}
	else {

	message( "Not using corba servers" )
	message( "If you want to use corba servers, set the variable")
	message( "OOCDIR to point to the path to corba")
	}
} else {
	message( "Compiling using corba servers" )
	message( "Ensure that this is the correct Corba path: $${MYOOCDIR}")
	DEFINES += USINGCORBA
	unix:INCLUDEPATH += $${MYOOCDIR}/include $${MYOOCDIR}/ob/include $${MYOOCDIR}/naming/include
	unix:LIBS += -L$${MYOOCDIR}/lib -lCosNaming -lOB
}	
 
win32 {
	DEFINES += WIN32
}

macx {
	DEFINES += __APPLE__
}

