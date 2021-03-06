# qmake project generated by QMsDev
#
# General settings

TEMPLATE = lib
CONFIG += warn_off staticlib create_prl exceptions rtti
CONFIG -= qt
TARGET  += PocketTunnel

INCLUDEPATH += ../VolumeWidget ../libLBIE ../GeometryFileTypes
QMAKE_CXXFLAGS += $$(CPPFLAGS)

DEFINES += HUGE=1000000

# Input

SOURCES =  \
		pocket_tunnel.cpp \
		handle.cpp \
		hfn_util.cpp \
		init.cpp \
		intersect.cpp \
		op.cpp \
		rcocone.cpp \
		robust_cc.cpp \
		smax.cpp \
		tcocone.cpp \
		util.cpp

HEADERS =  \
		datastruct.h \
		tcocone.h \
		rcocone.h \
		util.h \
		init.h \
		smax.h \
		pocket_tunnel.h \
		handle.h

