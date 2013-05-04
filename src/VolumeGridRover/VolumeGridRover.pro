TEMPLATE	= lib
LANGUAGE	= C++

CONFIG	+= create_prl qt staticlib opengl warn_on thread rtti exceptions

DEFINES	+= _FILE_OFFSET_BITS=64 USING_VOLUMEGRIDROVER_MEDAX MEDAX_INSERT_EDGES MEDAX_ALWAYS_INSERT

INCLUDEPATH	+= ../XmlRPC ../QGLViewer ../glew ../Segmentation/GenSeg ../VolMagick ../Tiling ../VolumeWidget ../GeometryFileTypes ../libLBIE

SOURCES	+= VolumeGridRover.cpp PointClassFile.cpp ContourFile.cpp SDF2D.cpp bspline_opt.cpp bspline_fit.cpp sdf_opt.cpp

HEADERS	+= VolumeGridRover.h SurfRecon.h PointClassFile.h ContourFile.h EM.h SDF2D.h bspline_opt.h sdf_opt.h medax.h

FORMS	= VolumeGridRoverBase.ui

TARGET = VolumeGridRover

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

linux-g++ | linux-g++-64 {
DEFINES += __LINUX__
}

win32 {
DEFINES += __WIN32__ __WINDOWS__ QGLVIEWER_STATIC WIN32
}

win32-g++ {
DEFINES += USING_GCC
}

freebsd | netbsd | openbsd | macx {
DEFINES += __BSD__
}

solaris-cc | solaris-cc-64 | solaris-g++ | solaris-g++-64 {
DEFINES += __SOLARIS__
} 

debug {
	DEFINES += DEBUG
}

macx {
        DEFINES += __APPLE__
        CONFIG += create_prl opengl
}

profiled {
  linux-g++ | linux-g++-64 | macx-g++ {
        QMAKE_CFLAGS+=-pg
        QMAKE_CXXFLAGS+=-pg
  }
}

#contains( DEFINES, USING_PE_DETECTION ) {
#	INCLUDEPATH += ../PEDetection
#}
