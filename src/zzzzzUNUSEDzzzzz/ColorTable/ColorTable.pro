TEMPLATE = lib
CONFIG += create_prl qt warn_off staticlib
TARGET  += ColorTable

INCLUDEPATH += ../libcontourtree

# Input

SOURCES =  \
		AlphaMap.cpp \
		ColorMap.cpp \
		ColorTable.cpp \
		ColorTableInformation.cpp \
		IsocontourMap.cpp \
		ConTreeMap.cpp \
		ContourSpectrumInfo.cpp \
		XoomedIn.cpp \
		XoomedOut.cpp

HEADERS =  \
		AlphaMap.h \
		ColorMap.h \
		ColorTable.h \
		ColorTableInformation.h \
		IsocontourMap.h \
		ConTreeMap.h \
		ContourSpectrumInfo.h \
		XoomedIn.h \
		XoomedOut.h

