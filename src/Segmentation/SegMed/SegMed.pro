TEMPLATE = lib 
CONFIG += create_prl warn_on release staticlib
CONFIG -= qt
TARGET += SegMed 

INCLUDEPATH += . ../../XmlRPC

linux-g++ | macx-g++ {
	QMAKE_CFLAGS += -funroll-loops -fomit-frame-pointer
	QMAKE_LFLAGS += -static
}

win32 {
	DEFINES += __WINDOWS__
}

HEADERS = segmed.h
SOURCES = GVF.cpp inout.cpp segment.cpp diffuse.cpp medical_segmentation.cpp

profiled {
  linux-g++ | linux-g++-64 | macx-g++ {
        QMAKE_CFLAGS+=-pg
        QMAKE_CXXFLAGS+=-pg
  }
}
