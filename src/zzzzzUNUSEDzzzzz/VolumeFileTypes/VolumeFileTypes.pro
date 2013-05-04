TEMPLATE = lib
CONFIG += create_prl qt warn_off staticlib exceptions rtti
#CONFIG += create_prl debug
TARGET  += VolumeFileTypes
INCLUDEPATH += ../ByteOrder ../Filters ../dataCutterClient \
    ../libcontour ../libcontourtree ../Volume

contains( QMAKE_CXXFLAGS_RELEASE, -fno-exceptions ) {
        # have to enable exceptions
        QMAKE_CXXFLAGS_RELEASE += -fexceptions
}

# unix only kludge to get around 4GB limit in Qt 3
unix {
	DEFINES += LARGEFILE_KLUDGE
	SOURCES += pfile.cpp
	HEADERS += pfile.h
}

solaris-g++ | solaris-g++-64 | solaris-cc | solaris-cc-64 {
	DEFINES += SOLARIS
}

win32-g++ {
	DEFINES += USING_GCC
}

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

# Input

SOURCES +=  \
		BasicVolumeFileImpl.cpp \
		DataCutterSource.cpp \
		DownLoadManager.cpp \
		MrcFileImpl.cpp \
		PifFileImpl.cpp \
		RawIVFileImpl.cpp \
		RawIVSimpleSource.cpp \
		RawVFileImpl.cpp \
		SourceManager.cpp \
		VolumeBuffer.cpp \
		VolumeBufferManager.cpp \
		VolumeFile.cpp \
		VolumeFileFactory.cpp \
		VolumeFileSource.cpp \
		VolumeFileSink.cpp \
		VolumeTranscriber.cpp \
		VolumeSource.cpp \
		VolumeSink.cpp \

HEADERS +=  \
		BasicVolumeFileImpl.h \
		DataCutterSource.h \
		DownLoadManager.h \
		MrcFileImpl.h \
		PifFileImpl.h \
		RawIVFileImpl.h \
		RawVFileImpl.h \
		SourceManager.h \
		VolumeBuffer.h \
		VolumeBufferManager.h \
		VolumeFileFactory.h \
		VolumeFile.h \
		VolumeFileSource.h \
		VolumeFileSink.h \
		VolumeTranscriber.h \
		VolumeSource.h \
		VolumeSink.h
		


