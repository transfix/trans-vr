TEMPLATE = lib 
CONFIG += warn_on staticlib create_prl rtti exceptions
CONFIG -= qt
TARGET = Tiling

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

INCLUDEPATH += ../VolumeGridRover ../VolumeWidget ../VolMagick ../libLBIE ../GeometryFileTypes

DEFINES += __HAVE_BOOST__

HEADERS = \
	approximate.h \
	branch.h \
	branch_util.h \
	common.h \
	contour.h \
	contour_read.h \
	correspond.h \
	cross.h \
	ct/ct.h \
	ct/slc.h \
	ct/slice.h \
	ct/sunras.h \
	ct/tiling.h \
	cubes.h \
	dec_type.h \
	decom_util.h \
	decompose.h \
	filter.h \
	gen_data.h \
	group.h \
	image2cont2D.h \
	legal_region.h \
	libHead.h \
	math_util.h \
	mymarch.h \
	myutil.h \
	parse_config.h \
	qsort.h \
	read_slice.h \
	scale.h \
	tile_hash.h \
	tile_util.h \
	tiling_main.h \
	tiling.h \
	SeriesFileReader.h

SOURCES = \
	CTSliceH2DInit.cpp \
	CTSliceHartley2D.cpp \
	CTVolumeHartley3D.cpp \
	#Libmain.cpp \
	#Libparse_config.cpp \
	approximate.cpp \
	branch.cpp \
	branch_util.cpp \
	contour.cpp \
	contour_read.cpp \
	correspond.cpp \
	cross.cpp \
	#ct/tiling.cpp \
	decom_util.cpp \
	decompose.cpp \
	filter.cpp \
	gen_data.cpp \
	group.cpp \
	image2cont2D.cpp \
	legal_region.cpp \
	main.cpp \
	math_util.cpp \
	mymarch.cpp \
	myutil.cpp \
	parse_config.cpp \
	qsort.cpp \
	read_slice.cpp \
	scale.cpp \
	slice.cpp \
	tile_hash.cpp \
	tile_util.cpp \
	tiling_main.cpp \
	volume.cpp \
	tiling.cpp \
	SeriesFileReader.cpp

win32 {
	DEFINES += __WINDOWS__
}
