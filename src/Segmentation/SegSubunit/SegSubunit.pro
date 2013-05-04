TEMPLATE = lib 
CONFIG += warn_on release staticlib
CONFIG -= qt
LANGUAGE = CPP
TARGET += SegSubunit
INCLUDEPATH += ../../XmlRPC

linux-g++ {
        QMAKE_CFLAGS += -static
}

HEADERS = segsubunit.h
SOURCES = criticalPnts.cpp asym_segment.cpp local_symmetry.cpp covariance.cpp \
	  subunit_segment_refine.cpp subunit_segment.cpp get_average.cpp makematrix.cpp inout.cpp \
	  virus_segment_subunit.cpp

profiled {
  linux-g++ | linux-g++-64 | macx-g++ {
        QMAKE_CFLAGS+=-pg
        QMAKE_CXXFLAGS+=-pg
  }
}
