TEMPLATE = lib
#TEMPLATE = app
CONFIG  += warn_on staticlib rtti exceptions
CONFIG -= qt
TARGET  += VolMagick 

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

INCLUDEPATH += ../../inc

DEFINES += USING_VOLMAGICK_INR
DEFINES += USING_IMOD_MRC
#DEFINES += USING_HDF5

HEADERS = BoundingBox.h \
	Dimension.h \
	Exceptions.h \
	VolMagick.h \
	endians.h \
	VolumeCache.h \
	Null_IO.h \
	RawIV_IO.h \
	RawV_IO.h \
	MRC_IO.h \
	Spider_IO.h \
	VTK_IO.h

SOURCES = VolMagick.cpp \
	Null_IO.cpp \
	RawIV_IO.cpp \
	RawV_IO.cpp \ 
	MRC_IO.cpp \
	Spider_IO.cpp \
	VTK_IO.cpp \
	VolumeCache.cpp \
	AnisotropicDiffusion.cpp \
	BilateralFilter.cpp \
	ContrastEnhancement.cpp \
	GDTVFilter.cpp #main.cpp

contains( DEFINES, USING_VOLMAGICK_INR ) {
	SOURCES += INR_IO.cpp
	HEADERS += INR_IO.h
}

contains( DEFINES, USING_IMOD_MRC ) {
	DEFINES += NOTIFFLIBS
	SOURCES += IMOD_MRC_IO.cpp \
	libiimod/b3dutil.c \
	libiimod/diffusion.c \
	libiimod/iilikemrc.c \
	libiimod/iimage.c \
	libiimod/iimrc.c \
	libiimod/iitif.c \
	libiimod/ilist.c \
	libiimod/mrcfiles.c \
	libiimod/mrcsec.c \
	libiimod/mrcslice.c \
	libiimod/plist.c \
	libiimod/sliceproc.c \
	libiimod/tiffstub.c \
	libiimod/islice.c \
	libiimod/parallelwrite.c
	HEADERS += IMOD_MRC_IO.h
}

contains( DEFINES, USING_HDF5 ) {
	HEADERS += HDF5_IO.h
	SOURCES += HDF5_IO.cpp
}

solaris-cc | solaris-cc-64 | solaris-g++ | solaris-g++-64 {
DEFINES += __SOLARIS__
} 

win32-g++ | win32-msvc {
DEFINES += __WINDOWS__
}
