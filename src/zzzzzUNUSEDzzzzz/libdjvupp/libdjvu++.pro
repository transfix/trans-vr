TEMPLATE = lib
CONFIG += create_prl qt warn_off staticlib
TARGET  += djvu++ 

SOURCES = ByteStream.cpp \ 
	#GContainer.cpp \
	#GOS.cpp \ 
	#GString.cpp \ 
	ZPCodec.cpp \
	DjVuGlobal.cpp \
	GException.cpp \
	GSmartPointer.cpp \
	#GThreads.cpp

HEADERS = ByteStream.h DjVuGlobal.h GContainer.h GException.h GOS.h GSmartPointer.h GString.h GThreads.h ZPCodec.h
