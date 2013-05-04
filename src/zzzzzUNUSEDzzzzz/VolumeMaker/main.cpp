/*
  Copyright 2008 The University of Texas at Austin
  
	Authors: Jose Rivera <transfix@ices.utexas.edu>
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

/* $Id: main.cpp 1528 2010-03-12 22:28:08Z transfix $ */

#include <qapplication.h>
#include <VolumeMaker/VolumeMaker.h>

#include <VolumeMaker/VolMagickEventHandler.h>

int main(int argc, char **argv)
{
  QApplication app(argc,argv);
  VolumeMaker vm;

  app.setMainWidget(&vm);
  vm.show();

  //VolMagickEventBasedOpStatus status;
  VolMagickOpStatus status;
  VolMagick::setDefaultMessenger(&status);

  return app.exec();
}
