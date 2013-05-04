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

/* $Id: VolumeInterface.h 1527 2010-03-12 22:10:16Z transfix $ */

#ifndef __VOLUMEINTERFACE_H__
#define __VOLUMEINTERFACE_H__

#include "volumeinterfacebase.h"

#include <VolMagick/VolMagick.h>

class VolumeInterface : public VolumeInterfaceBase
{
  Q_OBJECT

 public:
  VolumeInterface( const VolMagick::VolumeFileInfo &vfi,
		   QWidget* parent = 0, const char* name = 0, WFlags f = WType_TopLevel );
  ~VolumeInterface();

  void setInterfaceInfo(const VolMagick::VolumeFileInfo &vfi);

  void dimensionModifySlot();
  void boundingBoxModifySlot();

  void addTimestepSlot();
  void addVariableSlot();
  void deleteTimestepSlot();
  void deleteVariableSlot();
  void editVariableSlot();
  void importDataSlot();
  void remapSlot();

  VolMagick::VolumeFileInfo _vfi;

 private:
  void getSelectedVarTime(int &var, int &time);
};

#endif
