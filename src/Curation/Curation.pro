TEMPLATE = lib

CONFIG	+= warn_on staticlib create_prl rtti exceptions
CONFIG  -= qt

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

INCLUDEPATH += ../VolumeWidget ../libLBIE ../GeometryFileTypes

TARGET  += Curation 
LIBS += -lCGAL

HEADERS = \
./util.h \
./smax.h \
./robust_cc.h \
./op.h \
./mesh_io.h \
./medax.h \
./mds.h \
./intersect.h \
./init.h \
./hfn_util.h \
./datastruct.h \
./am.h \
Curation.h

SOURCES = \
./util.cpp \
./smax.cpp \
./robust_cc.cpp \
./op.cpp \
./mesh_io.cpp \
./medax.cpp \
./main.cpp \
./intersect.cpp \
./init.cpp \
./hfn_util.cpp \
./am.cpp
