# qmake project generated by QMsDev
#
# General settings

TEMPLATE = app
CONFIG += create_prl qt warn_off link_prl opengl
TARGET  += AnimationMaker
INCLUDEPATH += ../GeometryFileTypes ../VolumeWidget ../VolumeLibrary ../ByteOrder \
	../VolumeFileTypes ../Filters ../Contouring ../libLBIE
LIBS += ../GeometryFileTypes/libGeometryFileTypes.a ../ByteOrder/libByteOrder.a \
		../VolumeWidget/libVolumeWidget.a ../VolumeLibrary/libVolumeLibrary.a \
		../VolumeFileTypes/libVolumeFileTypes.a \
		../Contouring/libContouring.a ../libcontour/libcontour.linux.a \
		../c2c_codec/libc2c_codec.a ../libcontourtree/libcontourtree.linux.a

# Input

SOURCES =  \
		Animation.cpp \
		AnimationNode.cpp \
		AnimationWidget.cpp \
		GeometryRenderer.cpp \
		main.cpp \
		MouseSliderHandler.cpp \
		RawIVTestRenderable.cpp \
		RoverRenderable.cpp \
		ViewState.cpp

HEADERS =  \
		Animation.h \
		AnimationNode.h \
		AnimationWidget.h \
		GeometryRenderer.h \
		MouseSliderHandler.h \
		RawIVTestRenderable.h \
		RoverRenderable.h \
		ViewState.h

