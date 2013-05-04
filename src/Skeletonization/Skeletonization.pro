TEMPLATE        = lib
LANGUAGE        = C++

CONFIG  += warn_on staticlib create_prl rtti exceptions
CONFIG  -= qt

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

INCLUDEPATH += ../VolumeWidget ../libLBIE ../GeometryFileTypes

HEADERS += datastruct.h \
	 degen.h \
	 graph.h \
	 hfn_util.h \
	 init.h \
	 intersect.h \
	 medax.h \
	  op.h \
	 rcocone.h \
	 robust_cc.h \
	 skel.h \
	 tcocone.h \
	 u1.h \
	 u2.h \
	 util.h \
	 PolyTess.h

SOURCES +=degen.cpp  \
	graph.cpp \
	 hfn_util.cpp \
	 init.cpp  \
	intersect.cpp \
	 medax.cpp \
	 Skeletonization.cpp \
	 op.cpp  \
	rcocone.cpp \
	 robust_cc.cpp \
	 skel.cpp \
	 tcocone.cpp \
	 u1.cpp \
	 u2.cpp \
	 util.cpp \
	 PolyTess.cpp

TARGET  += Skeletonization

