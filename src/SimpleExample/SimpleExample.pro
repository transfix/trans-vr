TEMPLATE = app
CONFIG += create_prl warn_off opengl
TARGET  += SimpleExample
INCLUDEPATH += ../VolumeLibrary
unix:LIBS += ../VolumeLibrary/libVolumeLibrary.a \
                -lglut -lXi

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

# Input

SOURCES =  \
		SimpleExample.cpp

HEADERS =  \
		SimpleExample.h

