# qmake project generated by QMsDev
#
# General settings

TEMPLATE = lib
CONFIG  += qt warn_off staticlib opengl create_prl
TARGET  += HLevelSet


QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

#INCLUDEPATH += ../Blurmaps
#INCLUDEPATH += ../PDBParser
#INCLUDEPATH += ../UsefulMath

INCLUDEPATH += ../VolMagick



# Input

SOURCES =     HLevelSet_Recon2.cpp \
              #HLevelSet_Gauss.cpp \ 
               levelset_2_June5.cpp \
		#HLevelSet.cpp \      
		HLevelSet_SES.cpp \
		#HLevelSet_Recon.cpp 

HEADERS = \ 
		HLevelSet.h \
                Misc.h 
