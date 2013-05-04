TEMPLATE = lib
CONFIG += create_prl qt warn_off staticlib rtti exceptions
TARGET  += GeometryFileTypes

INCLUDEPATH += ../VolumeWidget ../c2c_codec ../arithlib ../libdjvu++ ../libLBIE

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

SOURCES =  \
		GeometryFileType.cpp \
		GeometryLoader.cpp \
		RawcFile.cpp \
		RawFile.cpp \
		RawncFile.cpp \
		RawnFile.cpp \
		C2CFile.cpp \
		ObjFile.cpp \
		PcdFile.cpp \
		PcdsFile.cpp \
		LineFile.cpp \
		LinecFile.cpp	

HEADERS =  \
		GeometryFileType.h \
		GeometryLoader.h \
		RawcFile.h \
		RawFile.h \
		RawncFile.h \
		RawnFile.h \
		C2CFile.h \
		ObjFile.h \
		PcdFile.h \
		PcdsFile.h \
		LineFile.h \
		LinecFile.h \
		cvcraw_geometry.h
