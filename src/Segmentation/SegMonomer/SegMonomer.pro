TEMPLATE = lib
CONFIG += create_prl warn_on release staticlib
CONFIG -= qt
LANGUAGE = CPP
TARGET += SegMonomer
INCLUDEPATH += ../../XmlRPC

linux-g++ {
        QMAKE_CFLAGS += -static
}

HEADERS = segmonomer.h
SOURCES = gvf.cpp monomer_segment.cpp inout.cpp virus_segment_monomer.cpp

profiled {
  linux-g++ | linux-g++-64 | macx-g++ {
        QMAKE_CFLAGS+=-pg
        QMAKE_CXXFLAGS+=-pg
  }
}
