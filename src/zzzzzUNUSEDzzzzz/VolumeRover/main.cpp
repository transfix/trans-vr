/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
	         Jose Rivera <transfix@cs.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Volume Rover.

  Volume Rover is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  Volume Rover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/


#include <qapplication.h>
#include <qstring.h>
#include <q3textedit.h>
//#include <qguardedptr.h>
#include <q3progressdialog.h>
#include <qevent.h>
#include <VolumeRover/newvolumemainwindow.h>
#include <VolumeRover/terminal.h>
#include <VolumeRover/VolMagickEventHandler.h>
#include <glew/glew.h>

#ifdef Q_OS_UNIX
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif

/*
#include <VolumeWidget/SimpleOpenGLWidget.h>
#include <VolumeWidget/ZoomInteractor.h>
#include <VolumeWidget/TrackballRotateInteractor.h>
#include <VolumeWidget/RotateInteractor.h>
#include <VolumeWidget/PanInteractor.h>
#include <VolumeWidget/WorldAxisRotateInteractor.h>
#include <VolumeWidget/WireCubeRenderable.h>
#include <VolumeWidget/TransformRenderable.h>
#include <VolumeWidget/RenderableArray.h>
#include <VolumeWidget/VolumeRenderable.h>
#include <VolumeWidget/Mouse3DAdapter.h>
//#include <MySplitter.h>
//#include "IPolyRenderable.h"
#include <VolumeRover/Rover3DWidget.h>
*/

QPointer<NewVolumeMainWindow> mainwindow;

void msgHandler(QtMsgType type, const char *msg)
{
  QString qmsg(msg);
  if(mainwindow && mainwindow->getTerminal()) 
    mainwindow->getTerminal()->text->append(qmsg);
  printf("%s\n",msg);
}

int main( int argc, char** argv )
{
#if defined(Q_OS_UNIX) && defined(USING_PE_DETECTION)
  //increase the stack size for PEDetection!
   struct rlimit MaxLimit;
	
   getrlimit(RLIMIT_STACK, &MaxLimit);
   printf ("Current stack Size: \n");
   printf ("rlim_cur = %d, ", (int)MaxLimit.rlim_cur);
   printf ("rlim_max = %d\n", (int)MaxLimit.rlim_max);
   fflush (stdout);
   
   MaxLimit.rlim_cur = 1024*1024*32;
   MaxLimit.rlim_max = 1024*1024*128;
   
   setrlimit(RLIMIT_STACK, &MaxLimit);
   
   getrlimit(RLIMIT_STACK, &MaxLimit);
   printf ("Increased stack Size: \n");
   printf ("rlim_cur = %d, ", (int)MaxLimit.rlim_cur);
   printf ("rlim_max = %d\n", (int)MaxLimit.rlim_max);
   fflush (stdout);
#endif   

   QApplication::setColorSpec( QApplication::ManyColor );
   QApplication app( argc, argv );
   int retval;
   
   mainwindow = new NewVolumeMainWindow;
   app.setMainWidget(mainwindow);
   qInstallMsgHandler(msgHandler);
   mainwindow->show();
   mainwindow->init(); //set up GL scene

   VolMagickEventBasedOpStatus status;
   VolMagick::setDefaultMessenger(&status);

   retval = app.exec();
   delete (NewVolumeMainWindow*) mainwindow; /* make sure destructor is called in case glDeleteTextures is called or something */
   
   return retval;
}

