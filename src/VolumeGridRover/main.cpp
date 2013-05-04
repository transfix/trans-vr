/*
  Copyright 2005-2006 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeGridRover.

  VolumeGridRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeGridRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <qapplication.h>
#include <qstring.h>

#ifndef WIN32
#include <signal.h>
#endif

#include <VolumeGridRover/VolumeGridRoverMainWindow.h>

#include <glew/glew.h>

int main(int argc, char **argv)
{
  int retval;
  
#ifndef WIN32
  struct sigaction mysigaction;
    
  sigemptyset(&mysigaction.sa_mask);
  mysigaction.sa_flags = 0;
  mysigaction.sa_handler = SIG_IGN;
  sigaction(SIGPIPE, &mysigaction, NULL);
#endif
  
  QApplication::setColorSpec(QApplication::ManyColor);
  QApplication app(argc,argv);
  
  VolumeGridRoverMainWindow *mainwindow = new VolumeGridRoverMainWindow;
  app.setMainWidget(mainwindow);
  mainwindow->show();
  retval = app.exec();
  delete mainwindow; /* make sure the destructor for mainwindow is called before
				        CVCGL_Shutdown, because GL is still used in SliceCanvas's
						destructor (calls glDeleteTextures) */
  
  return retval;
}
