TEMPLATE	= app
LANGUAGE	= C++

CONFIG	+= qt warn_on thread release windows rtti exceptions

LIBS	+= ../VolMagick/libVolMagick.a -lboost_regex -lboost_filesystem -lboost_system

INCLUDEPATH	+= ../VolMagick

HEADERS	+= VolumeMaker.h \
	VolMagickEventHandler.h \
	VolumeInterface.h \
	NewVolumeDialog.h \
	DimensionModify.h \
	BoundingBoxModify.h \
	ImportData.h \
	RemapVoxels.h

SOURCES	+= main.cpp \
	VolumeMaker.cpp \
	VolumeInterface.cpp \
	NewVolumeDialog.cpp \
	DimensionModify.cpp \
	BoundingBoxModify.cpp \
	ImportData.cpp \
	RemapVoxels.cpp

FORMS	= volumemakerbase.ui \
	volumeinterfacebase.ui \
	addvariablebase.ui \
	addtimestepbase.ui \
	importdatabase.ui \
	newvolumedialogbase.ui \
	dimensionmodifybase.ui \
	boundingboxmodifybase.ui \
	editvariablebase.ui \
	remapvoxelsbase.ui

#win32:CONFIG += console






TARGET          = VolumeMaker


QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

contains( DEFINES, USING_HDF5 ){
     LIBS += -lhdf5_cpp -lhdf5
}

#unix {
#  UI_DIR = .ui
#  MOC_DIR = .moc
#  OBJECTS_DIR = .obj
#}


