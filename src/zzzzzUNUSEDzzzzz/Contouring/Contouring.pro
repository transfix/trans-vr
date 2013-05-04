TEMPLATE = lib
CONFIG += create_prl qt warn_off staticlib opengl rtti exceptions
TARGET  += Contouring
INCLUDEPATH += ../glew ../VolumeWidget ../duallib ../libLBIE ../GeometryFileTypes

win32 {
	DEFINES += WIN32
}

macx {
	DEFINES += __APPLE__
}

#DEFINES += DUALLIB

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

SOURCES =  \
		Contour.cpp \
		ContourExtractor.cpp \
		ContourGeometry.cpp \
		MarchingCubesBuffers.cpp \
		MultiContour.cpp \
		RGBAExtractor.cpp \
		SingleExtractor.cpp 

HEADERS =  \
		Contour.h \
		ContourExtractor.h \
		ContourGeometry.h \
		cubes.h \
		MarchingCubesBuffers.h \
		MultiContour.h \
		RGBAExtractor.h \
		SingleExtractor.h

