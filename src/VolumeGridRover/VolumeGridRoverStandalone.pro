TEMPLATE        = app
LANGUAGE	= C++

CONFIG += create_prl qt warn_on release opengl thread

unix:LIBS	+= ../XmlRPC/libXmlRPC.a ../ColorTable/libColorTable.a ../QGLViewer/libQGLViewer.a ../glew/libglew.a ../Segmentation/GenSeg/libGenSeg.a ../PEDetection/libPEDetection.a

DEFINES	+= _FILE_OFFSET_BITS=64 USING_QT #EM_CLUSTERING

INCLUDEPATH	+= ../XmlRPC ../ColorTable ../libcontourtree ../QGLViewer ../glew ../Segmentation/GenSeg

HEADERS	+= \ 
	VolumeGridRover.h \
	MappedRawVFile.h \
	MappedRawIVFile.h \
	MappedVolumeFile.h \
	lfmap.h \
	VolumeGridRoverMainWindow.h

SOURCES	+= \ 
	VolumeGridRover.cpp \
	MappedRawVFile.cpp \
	MappedRawIVFile.cpp \
	MappedVolumeFile.cpp \
	lfmap.c \
	VolumeGridRoverMainWindow.cpp \
	main.cpp

FORMS	= VolumeGridRoverBase.ui \
	VolumeGridRoverMainWindowBase.ui

TARGET = VolumeGridRover

linux-g++ | linux-g++-64 {
DEFINES += __LINUX__
}
 
win32 {
DEFINES += __WIN32__ __WINDOWS__ WIN32
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
        DEFINES += DYNAMIC_GL __APPLE__
        CONFIG += create_prl opengl
}

dynamic_gl {
        DEFINES += DYNAMIC_GL
        CONFIG -= opengl
        QMAKE_LFLAGS = -rdynamic
}

contains( DEFINES, EM_CLUSTERING ) {
	INCLUDEPATH += ../PEDetection
}
