TEMPLATE = lib
CONFIG += create_prl qt warn_off staticlib
TARGET  += PEDetection 

# Input

INCLUDEPATH += ../VolMagick

SOURCES =  main.cpp Control.cpp GVF.cpp Initialization.cpp EM.cpp Geometric.cpp \
           TFGeneration.cpp Evaluation.cpp Stack.cpp PEDetection.cpp \
           MarchingCubes.cpp Skeleton.cpp Thinning.cpp  \
           FrontPlane.cpp MembraneSeg.cpp Octree.cpp OctreeCell.cpp \
           VesselSeg.cpp CriticalPoints.cpp STree.cpp

HEADERS = \
CC_Geom.h               FrontPlane.h            MarchingCubes.h         Skeleton_SeedPts.h      VesselSeg.h \
CompileOptions.h        GVF.h                   SphereIndex.h           VesselSeg2D.h \
ConnV_AmbiF_Table.h     Geometric.h             MembraneSeg.h           Stack.h                \ 
Control.h               Initialization.h        Octree.h                TFGeneration.h \
CriticalPoints.h        MC_Configuration.h      PEDetection.h           Thinning.h \
EM.h                    MC_Geom.h               STree.h                 ThinningTables.h \
Evaluation.h            MC_Geom_Octree.h        Skeleton.h              Timer.h

macx-g++ {
	#QMAKE_CXXFLAGS = -frepo
}

#the build is broken on windows
win32 {
	HEADERS = 
	SOURCES = 
}
