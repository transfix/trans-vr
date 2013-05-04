TEMPLATE = app 
CONFIG += warn_on release
CONFIG -= qt
LANGUAGE = CPP
TARGET += SegServ 

INCLUDEPATH += ../GenSeg ../SegCapsid ../SegMed ../SegMonomer ../SegSubunit ../SecStruct ../../XmlRPC ../../VolMagick
unix:LIBS += ../GenSeg/libGenSeg.a ../SegCapsid/libSegCapsid.a ../SegMed/libSegMed.a \
	../SegMonomer/libSegMonomer.a ../SegSubunit/libSegSubunit.a ../SecStruct/libSecStruct.a ../../XmlRPC/libXmlRPC.a ../../VolMagick/libVolMagick.a -lm

g++ {
	QMAKE_CFLAGS += -funroll-loops -fomit-frame-pointer
	QMAKE_LFLAGS += -static
}

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

solaris-g++ | solaris-g++-64 {
	LIBS+=-lnsl -lsocket
}

win32 {
	DEFINES += __WINDOWS__
}

SOURCES = SegServ.cpp 

profiled {
  linux-g++ | linux-g++-64 | macx-g++ {
        QMAKE_CFLAGS+=-pg
        QMAKE_CXXFLAGS+=-pg
  }
}
