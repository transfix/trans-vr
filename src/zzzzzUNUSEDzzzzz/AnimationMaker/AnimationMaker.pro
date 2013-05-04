# General settings

TEMPLATE = lib
CONFIG += create_prl create_prl qt warn_off staticlib
TARGET  += Animation
INCLUDEPATH += ../VolumeWidget ../libLBIE

# Input

SOURCES =  \
		Animation.cpp \
		AnimationNode.cpp \
		ViewState.cpp

HEADERS =  \
		Animation.h \
		AnimationNode.h \
		ViewState.h

