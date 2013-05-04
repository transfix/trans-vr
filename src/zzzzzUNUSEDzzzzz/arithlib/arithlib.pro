TEMPLATE = lib
CONFIG += create_prl qt warn_off staticlib
TARGET  += arith 
INCLUDEPATH += ../libdjvu++

SOURCES = arithdecode.cpp bitbuffer.cpp  coder.cpp  encode.cpp stats.cpp  utils.cpp

HEADERS = arith.h arith_defines.h bitbuffer.h  coder.h  encode.h stats.h  unitypes.h  utils.h 
