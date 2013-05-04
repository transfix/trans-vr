TEMPLATE = lib
CONFIG += create_prl qt warn_off staticlib
TARGET  += RenderServers
INCLUDEPATH += ./ ../ColorTable ../libcontourtree ../VolumeWidget ../libLBIE

contains( QMAKE_CXXFLAGS_RELEASE, -fno-exceptions ) {
	# have to enable exceptions
	QMAKE_CXXFLAGS_RELEASE += -fexceptions
}


# Input

SOURCES =  \
		FrameInformation.cpp \
		raycastserversettingsdialogimpl.cpp \
		RenderServer.cpp \
		textureserversettingsdialogimpl.cpp \
		TransferArray.cpp \

HEADERS =  \
		FrameInformation.h \
		raycastserversettingsdialogimpl.h \
		RenderServer.h \
		textureserversettingsdialogimpl.h \
		TransferArray.h \

FORMS =  \
		textureserversettingsdialog.ui \
		raycastserversettingsdialog.ui

QMAKE_CXXFLAGS += $$(CPPFLAGS)
QMAKE_LFLAGS += $$(LDFLAGS)
		
MYOOCDIR = $$(OOCDIR)
#MYOOCDIR =
isEmpty(MYOOCDIR) : !contains(DEFINES, USING_CORBA) {
	message( "Not using corba servers" )
	message( "If you want to use corba servers, set the variable")
	message( "OOCDIR to point to the path to corba")
} else {

	message( "Compiling using corba servers" )
	!contains(DEFINES, USING_CORBA) {
		message( "Ensure that this is the correct Corba path: $${MYOOCDIR}")
        }

	# compiling idls
	message( "Compiling the idl's")
	system(idl --no-skeletons 3DTex.idl) {
		#error( "An error occured while compiling 3DTex.idl")
		SUCCESS += 3DTex.idl
	}
	system(idl --no-skeletons pvolserver.idl) {
		#error( "An error occured while compiling pvolserver.idl")
		SUCCESS += pvolserver.idl
	}
	system(idl --no-skeletons cr.idl) {
		#error( "An error occured while compiling cr.idl")
		SUCCESS += cr.idl
	}

	isEmpty( SUCCESS ) {
		error("Could not compile the required idl's ")
	}

	!contains(DEFINES, USING_CORBA) {
		unix:INCLUDEPATH += $${MYOOCDIR}/include $${MYOOCDIR}/ob/include $${MYOOCDIR}/naming/include
	}
	SOURCES +=  \
			3DTex.cpp \
			cr.cpp \
			Corba.cpp \
			pvolserver.cpp \
			RaycastRenderServer.cpp \
			TextureRenderServer.cpp \
			IsocontourRenderServer.cpp
	
	HEADERS +=  \
			3DTex.h \
			cr.h \
			Corba.h \
			pvolserver.h \
			RaycastRenderServer.h \
			RenderServer.h \
			TextureRenderServer.h \
			IsocontourRenderServer.h
}
