TEMPLATE = lib 
CONFIG += create_prl warn_on release staticlib
CONFIG -= qt
LANGUAGE = C
TARGET += GenSeg

INCLUDEPATH += . ../../XmlRPC ../../VolMagick

linux-g++ | macx-g++ {
	QMAKE_CFLAGS += -funroll-loops -fomit-frame-pointer
	QMAKE_LFLAGS += -static
}

win32 {
	DEFINES += __WINDOWS__
}

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

HEADERS = genseg.h
SOURCES = inout.cpp segment.cpp general_segmentation.cpp

profiled {
  linux-g++ | linux-g++-64 | macx-g++ {
        QMAKE_CFLAGS+=-pg
        QMAKE_CXXFLAGS+=-pg
  }
}
