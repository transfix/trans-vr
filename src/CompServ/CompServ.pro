TEMPLATE = app 
CONFIG += link_prl warn_on console
CONFIG -= qt
LANGUAGE = CPP
TARGET += CompServ 

#DEFINES += USING_PE_DETECTION

#Sangmins Pulmonary Embolus Detection code
contains( DEFINES, USING_PE_DETECTION ) {
	LIBS += ../PEDetection/libPEDetection.a
	INCLUDEPATH += ../PEDetection
}

INCLUDEPATH += ../Segmentation/GenSeg ../Segmentation/SegCapsid ../Segmentation/SegMed ../Segmentation/SegMonomer \
	../Segmentation/SegSubunit ../Segmentation/SecStruct ../XmlRPC ../VolMagick

!win32 | win32-g++ {
LIBS += ../Segmentation/GenSeg/libGenSeg.a ../Segmentation/SegCapsid/libSegCapsid.a ../Segmentation/SegMed/libSegMed.a \
	../Segmentation/SegMonomer/libSegMonomer.a ../Segmentation/SegSubunit/libSegSubunit.a \
	../Segmentation/SecStruct/libSecStruct.a ../VolMagick/libVolMagick.a \
	../XmlRPC/libXmlRPC.a ../VolMagick/libVolMagick.a -lCGAL -lm -lboost_regex
win32:LIBS += -lwsock32
}

g++ {
	QMAKE_CFLAGS += -funroll-loops -fomit-frame-pointer
	QMAKE_LFLAGS += -static
}

solaris-g++ | solaris-g++-64 {
	LIBS+=-lnsl -lsocket
}

win32 {
	DEFINES += __WINDOWS__
}

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

SOURCES = CompServ.cpp 

profiled {
  linux-g++ | linux-g++-64 | macx-g++ {
        QMAKE_CFLAGS+=-pg
        QMAKE_CXXFLAGS+=-pg
  }
}

unix {
	DEFINES += __UNIX__
}

contains( DEFINES, USING_HDF5 ){
        LIBS += -lhdf5_cpp -lhdf5
}

