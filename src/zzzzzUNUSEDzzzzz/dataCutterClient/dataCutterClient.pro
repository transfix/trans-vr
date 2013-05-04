TEMPLATE = lib
CONFIG += create_prl qt warn_off staticlib
TARGET  += dataCutterClient 

SOURCES = byemessage.cpp \
	dsdetails.cpp \
	errormessage.cpp \
	getdsdetailsack.cpp \
	getdsdetailsmessage.cpp \
	getdsmessage.cpp \
	getdsmessageack.cpp \
	hellomessage.cpp \
	hellomessageack.cpp \
	#imageclient.cpp \ 
	message.cpp

HEADERS = dsentry.h       imageclient.h   message.h 
