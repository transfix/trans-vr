/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
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

#ifndef SIMPLEEXAMPLE_H
#define SIMPLEEXAMPLE_H

#define ROTATE 1
#define TRANSLATE 2

int BeginGraphics (int* argc, char** argv, const char* name);

void RegisterCallbacks();

void init();

void InitProjection();

void InitModelMatrix();

void InitLighting();

void InitState();

void InitVolumeRenderer();

void InitData();

void Display();

void Idle();

void MouseButton(int button, int mstate, int x, int y);

void MouseMove(int x, int y);

void Keyboard(unsigned char key, int x, int y);

void Rotate(int dx, int dy);

void Translate(int dx, int dy, int dz);



#endif
