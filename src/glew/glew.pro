TEMPLATE        = lib
LANGUAGE	= C
TARGET += glew 

CONFIG += create_prl warn_off opengl release staticlib

HEADERS	+= glew.h wglew.h glxew.h

SOURCES	+= glew.c 

profiled {
  linux-g++ | linux-g++-64 | macx-g++ {
        QMAKE_CFLAGS+=-pg
        QMAKE_CXXFLAGS+=-pg
  }
}

