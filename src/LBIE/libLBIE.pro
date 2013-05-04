TEMPLATE = lib
CONFIG  += qt warn_on staticlib create_prl rtti exceptions
TARGET  += LBIE 

INCLUDEPATH += ../VolMagick ../libcontour ../FastContouring

contains( DEFINES, USING_LBIE_GEOFRAME_SDF_REFINEMENT )
{
# Volume -- for quality_improve.cpp methods for copying Geoframe to Geometry
# VolumeWidget -- for Geometry
# multi_sdf -- for SDF calculation
# GeometryFileTypes -- for GeometryLoader

	INCLUDEPATH += \
		../Volume \
		../VolumeWidget \
		../multi_sdf \
		../GeometryFileTypes
}

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)

# Input
HEADERS = \
		LBIE_Mesher.h \
                e_face.h \
                LBIE_geoframe.h \
                normalspline.h \
                octree.h \
                pcio.h

SOURCES = \
		LBIE_Mesher.cpp \
		e_face.cpp \
		LBIE_geoframe.cpp \
		hexa.cpp \
		normalspline.cpp \
		octree.cpp \
		pcio.cpp \
		tetra.cpp
