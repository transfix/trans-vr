TEMPLATE = lib
CONFIG += create_prl warn_off staticlib opengl
TARGET  += VolumeLibrary

INCLUDEPATH += ../glew

win32 {
	DEFINES += WIN32
}

macx {
	DEFINES += __APPLE__
}

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

#DEFINES += UPLOAD_DATA_RESIZE_HACK
#INCLUDEPATH += ../VolMagick

# Input

SOURCES =  \
		ClipCube.cpp \
		Extent.cpp \
		FragmentProgramARBImpl.cpp \
		FragmentProgramImpl.cpp \
		Paletted2DImpl.cpp \
		PalettedImpl.cpp \
		Plane.cpp \
		Polygon.cpp \
		PolygonArray.cpp \
		Renderer.cpp \
		RendererBase.cpp \
		RGBABase.cpp \
		SGIColorTableImpl.cpp \
		SimpleRGBA2DImpl.cpp \
		SimpleRGBAImpl.cpp \
		UnshadedBase.cpp \
		VolumeRenderer.cpp \
		VolumeRendererFactory.cpp 

HEADERS =  \
		ClipCube.h \
		ExtensionPointers.h \
		Extent.h \
		FragmentProgramARBImpl.h \
		FragmentProgramImpl.h \
		LookupTables.h \
		Paletted2DImpl.h \
		PalettedImpl.h \
		Plane.h \
		Polygon.h \
		PolygonArray.h \
		Renderer.h \
		RendererBase.h \
		RGBABase.h \
		SGIColorTableImpl.h \
		SimpleRGBA2DImpl.h \
		SimpleRGBAImpl.h \
		StaticExtensionPointers.h \
		UnshadedBase.h \
		VolumeRenderer.h \
		VolumeRendererFactory.h 

