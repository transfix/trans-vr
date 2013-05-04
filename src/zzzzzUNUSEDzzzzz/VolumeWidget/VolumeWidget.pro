TEMPLATE = lib
CONFIG += create_prl qt warn_off staticlib opengl 
TARGET  += VolumeWidget
INCLUDEPATH += ../glew ../VolumeLibrary ../Contouring ./ ../libLBIE ../GeometryFileTypes ../Volume ../VolMagick ../libcontour

win32 {
	DEFINES += WIN32
}

macx {
	DEFINES += __APPLE__
}

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

# Input

SOURCES =  \
		Aggregate3DHandler.cpp \
		Extents.cpp \
		Geometry.cpp \
		GeometryRenderable.cpp \
		Grid.cpp \
		IntQueue.cpp \
		Matrix.cpp \
		Mouse3DAdapter.cpp \
		Mouse3DHandler.cpp \
		#MouseEvent3D.cpp \
		MouseEvent3DPrivate.cpp \
		MouseHandler.cpp \
		OrthographicView.cpp \
		PanInteractor.cpp \
		PerspectiveView.cpp \
		Quaternion.cpp \
		RawGeometry.cpp \
		Ray.cpp \
		Renderable.cpp \
		RenderableArray.cpp \
		RenderableView.cpp \
		RotateInteractor.cpp \
		SimpleOpenGLWidget.cpp \
		ScaleInteractor.cpp \
		TrackballRotateInteractor.cpp \
		#TransformRenderable.cpp \
		Tuple.cpp \
		Vector.cpp \
		View.cpp \
		ViewInformation.cpp \
		ViewInteractor.cpp \
		#VolumeRenderable.cpp \
		WireCube.cpp \
		WireCubeRenderable.cpp \
		WorldAxisRotateInteractor.cpp \
		ZoomInteractor.cpp

HEADERS =  \
		Aggregate3DHandler.h \
		ExpandableArray.h \
		Extents.h \
		Geometry.h \
		GeometryRenderable.h \
		Grid.h \
		IntQueue.h \
		Matrix.h \
		Mouse3DAdapter.h \
		Mouse3DHandler.h \
		MouseEvent3D.h \
		MouseHandler.h \
		OrthographicView.h \
		PanInteractor.h \
		PerspectiveView.h \
		Quaternion.h \
		RawGeometry.h \
		Ray.h \
		Renderable.h \
		RenderableArray.h \
		RenderableView.h \
		RotateInteractor.h \
		SimpleOpenGLWidget.h \
		ScaleInteractor.h \
		TrackballRotateInteractor.h \
		TransformRenderable.h \
		Tuple.h \
		Vector.h \
		View.h \
		ViewInformation.h \
		ViewInteractor.h \
		VolumeRenderable.h \
		WireCube.h \
		WireCubeRenderable.h \
		WorldAxisRotateInteractor.h \
		ZoomInteractor.h

