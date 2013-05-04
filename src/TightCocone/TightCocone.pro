TEMPLATE	= lib 
LANGUAGE	= C++

CONFIG	+= warn_on staticlib create_prl rtti exceptions
CONFIG  -= qt

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

INCLUDEPATH += ../VolMagick ../VolumeWidget ../libLBIE ../GeometryFileTypes

# gmp
#LIBS += -lgmpxx
		
HEADERS	+= medax.h \
	init.h \
	util.h \
	segment.h \
	op.h \
	tcocone.h \
	rcocone.h \
	robust_cc.h \
	datastruct.h
SOURCES	+=      tight_cocone.cpp \
		init.cpp \ 
		medax.cpp \ 
		op.cpp  \
		rcocone.cpp \
		robust_cc.cpp \ 
		tcocone.cpp \ 
		util.cpp \
                segment.cpp \
                GVF.cpp \
                inout.cpp

TARGET  += TightCocone 

LIBS += -lCGAL

