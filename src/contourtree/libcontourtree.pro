# General settings

TEMPLATE = lib
CONFIG += create_prl warn_off staticlib
TARGET  += contourtree

macx {
	DEFINES += MACOS_X
}

# Input

SOURCES =  \
		computeCT.cpp
		
HEADERS =  \
		cellQueue.h \
		computeCT.h \
		HeightField.h \
		unionfind.h

