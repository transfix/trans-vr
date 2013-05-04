TEMPLATE = lib
TARGET += XmlRPC 

CONFIG += create_prl warn_off release staticlib exceptions rtti

HEADERS	+= base64.h \
	XmlRpcException.h \
	XmlRpcServer.h \
	XmlRpcSource.h \
	XmlRpcClient.h \
	XmlRpc.h \
	XmlRpcServerMethod.h \
	XmlRpcUtil.h \
	XmlRpcDispatch.h \
	XmlRpcServerConnection.h \
	XmlRpcSocket.h \
	XmlRpcValue.h
 

SOURCES	+= XmlRpcClient.cpp \
	XmlRpcServer.cpp \
	XmlRpcSource.cpp \
	XmlRpcDispatch.cpp \
	XmlRpcServerMethod.cpp \
	XmlRpcUtil.cpp \
	XmlRpcServerConnection.cpp \
	XmlRpcSocket.cpp \
	XmlRpcValue.cpp

linux-g++ | linux-g++-64 {
DEFINES += __LINUX__
}

win32 {
DEFINES += __WIN32__ __WINDOWS__ _WINDOWS
}

freebsd-g++ | netbsd-g++ | openbsd-g++ | macx-g++ {
DEFINES += __BSD__
}
