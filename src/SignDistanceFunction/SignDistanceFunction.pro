TEMPLATE = lib
CONFIG  += warn_off staticlib create_prl
TARGET  += SignDistanceFunction

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

INCLUDEPATH += ../VolMagick ../VolumeWidget ../libLBIE ../GeometryFileTypes

SOURCES =  \
		common.cpp \
		compute.cpp \
		head.cpp \
		init.cpp \
		main.cpp \
		new_adjust.cpp \
		octree.cpp \
		propagate.cpp


HEADERS =  \
		common.h \
		head.h \
		sdfLib.h 

