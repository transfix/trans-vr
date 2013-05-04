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

/* $Id: VolumeMaker.h 1527 2010-03-12 22:10:16Z transfix $ */

#include <map>
#include <VolMagick/VolMagick.h>
#include "volumemakerbase.h"

class QWidgetStack;
class QListView;
class QListViewItem;
class QGridLayout;
class VolumeInterface;

class VolumeMaker : public VolumeMakerBase
{
  Q_OBJECT
   
 public:
  VolumeMaker( QWidget* parent = 0, const char* name = 0, WFlags f = WType_TopLevel );
  virtual ~VolumeMaker();

  void newVolumeSlot();
  void openVolumeSlot();
  void closeVolumeSlot();

 protected:
  void addVolumeToStack(const VolMagick::VolumeFileInfo &vfi);

 protected slots:
  void raiseSelected();

 protected:
  QWidgetStack *_volumeStack;
  QListView *_volumeStackList;
  QGridLayout *_layout;

  std::map<QListViewItem*,VolumeInterface*> _itemToInterface;
  std::map<VolumeInterface*,QListViewItem*> _interfaceToItem;
};
