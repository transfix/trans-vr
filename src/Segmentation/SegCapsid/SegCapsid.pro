TEMPLATE = lib
CONFIG += warn_on release staticlib
CONFIG -= qt
LANGUAGE = CPP
TARGET = SegCapsid
INCLUDEPATH += . ../../XmlRPC

linux-g++ {
	QMAKE_CFLAGS += -static
}

HEADERS = segcapsid.h
SOURCES = criticalPnts.cpp diffuse.cpp global_symmetry.cpp \
	  capsid_segment_simple.cpp capsid_segment_single.cpp capsid_segment_score.cpp \
	  capsid_segment_march.cpp capsid_segment_double.cpp inout.cpp virus_segment_capsid.cpp

profiled {
  linux-g++ | linux-g++-64 | macx-g++ {
        QMAKE_CFLAGS+=-pg
        QMAKE_CXXFLAGS+=-pg
  }
}
