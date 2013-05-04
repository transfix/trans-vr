TEMPLATE	= lib 
LANGUAGE	= C++

CONFIG	+= warn_off staticlib create_prl console exceptions rtti
CONFIG  -= qt

INCLUDEPATH += ../VolMagick ../VolumeWidget ../libLBIE ../GeometryFileTypes

QMAKE_CXXFLAGS += $$(CPPFLAGS) -DHUGE=1000000000
QMAKE_LFLAGS += $$(LDFLAGS)

HEADERS	+= priorityqueue.h \
	mesh_io.h \
	init.h \
	sdf.h \
	kdtree.h \
	robust_cc.h \
	op.h \
	dt.h \
	datastruct.h \
	util.h \
	tcocone.h \
	mds.h \
	matrix.h \
	rcocone.h \
	multi_sdf.h

SOURCES	+=      main.cpp \
		mesh_io.cpp \
		sdf.cpp \
		kdtree.cpp \
		matrix.cpp \
		dt.cpp \
		init.cpp \
		robust_cc.cpp \
		rcocone.cpp \
		tcocone.cpp \
		op.cpp \
		util.cpp

TARGET  += multi_sdf 

