TEMPLATE	= app
LANGUAGE	= C++
TARGET		= RawIVEditor

CONFIG += link_prl qt warn_on windows thread

LIBS	+= ../ByteOrder/libByteOrder.a

INCLUDEPATH	+= ../ByteOrder

HEADERS	+= rawiveditordialog.h

SOURCES	+= main.cpp \
	rawiveditordialog.cpp

FORMS	= rawiveditordialogbase.ui
