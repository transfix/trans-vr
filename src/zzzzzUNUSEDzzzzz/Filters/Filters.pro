TEMPLATE = lib
CONFIG += create_prl warn_on staticlib rtti exceptions
TARGET  += Filters

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

INCLUDEPATH += ../VolumeWidget ../libLBIE ../GeometryFileTypes

SOURCES =  \
		BilateralFilter.cpp \
		Filter.cpp \
		OOCFilter.cpp \
		OOCBilateralFilter.cpp \
                ContrastEnhancement.cpp \
		Smoothing.cpp

HEADERS =  \
		BilateralFilter.h \
		Filter.h \
		OOCFilter.h \
		OOCBilateralFilter.h \
                ContrastEnhancement.h \
		Smoothing.h \
		project_verts.h
