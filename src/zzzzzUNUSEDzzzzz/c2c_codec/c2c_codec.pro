TEMPLATE = lib
CONFIG += create_prl warn_off staticlib
TARGET  += c2c_codec
INCLUDEPATH += ../arithlib ../libdjvu++
DEFINES += ARITH_ENCODE ARITH_DECODE UNIQUE ZP_CODEC

linux-g++|linux-g++-64 {
     	QMAKE_CXXFLAGS += -fpermissive
     	QMAKE_CXXFLAGS_RELEASE += -fpermissive
}

# Input

SOURCES =  \
		c2c_codec.cpp \
		bufferedio.cpp \
		ContourGeom.cpp \
		util.cpp \
		rawslicefac.cpp \
		rawvslicefac.cpp \
		slicecache.cpp \
		slice.cpp \
		layer.cpp \
		vertex.cpp

HEADERS =  \
		c2c_codec.h \
		bufferedio.h \
		c2cbuf.h \
		ContourGeom.h \
		cubes.h \
		decode.h \
		diskio.h \
		filec2cbuf.h \
		util.h \
		vtkMarchingCubesCases.h \
		vertex.h

