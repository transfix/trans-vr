TEMPLATE = lib
CONFIG += create_prl warn_on release staticlib exceptions rtti
CONFIG -= qt
LANGUAGE = CPP
TARGET = SecStruct
INCLUDEPATH += . ../../XmlRPC

linux-g++ {
	QMAKE_CFLAGS += -static
}

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

SOURCES = diffuse.cpp gvf.cpp helixhunter.cpp inout.cpp main.cpp sheethunter.cpp tensor.cpp fit_cylinder.cpp

HEADERS = march.h secstruct.h datastruct.h fit_cylinder.h

profiled {
  linux-g++ | linux-g++-64 | macx-g++ {
        QMAKE_CFLAGS+=-pg
        QMAKE_CXXFLAGS+=-pg
  }
}
