TEMPLATE	= lib
LANGUAGE	= C++

CONFIG += create_prl qt warn_on release staticlib

HEADERS	+= ColorTable2D.h \
	TableCanvas.h \
	AlphaCanvas.h \
	ColorCanvas.h

SOURCES	+= ColorTable2D.cpp \
	TableCanvas.cpp \
	AlphaCanvas.cpp \
	ColorCanvas.cpp

