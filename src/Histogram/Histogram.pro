TEMPLATE = lib
CONFIG   = qt warn_off staticlib create_prl
TARGET   = Histogram


LANGUAGE = C++


INCLUDEPATH  = ../Geometry
INCLUDEPATH += ../UsefulMath
INCLUDEPATH += ../OpenGL_Viewer
INCLUDEPATH += ../DataManager/SecondaryStructureDataManager
INCLUDEPATH += ../DataManager
INCLUDEPATH += ../Histogram


HEADERS  = glcontrolwidget.h
HEADERS += glgear.h
HEADERS += histogram.h
HEADERS += histogram_data.h


SOURCES  = glcontrolwidget.cpp
SOURCES += histogram.cpp
SOURCES += histogram_data.cpp
           

