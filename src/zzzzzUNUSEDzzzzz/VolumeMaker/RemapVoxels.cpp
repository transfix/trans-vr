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

/* $Id: RemapVoxels.cpp 1527 2010-03-12 22:10:16Z transfix $ */

#include <qvalidator.h>
#include <qlineedit.h>
#include <qmessagebox.h>
#include <VolumeMaker/RemapVoxels.h>

RemapVoxels::RemapVoxels(QWidget* parent, const char* name, WFlags f)
  : RemapVoxelsBase(parent,name,f)
{
  QDoubleValidator* doublev = new QDoubleValidator(this);
  _minValue->setValidator(doublev);
  _maxValue->setValidator(doublev);
}

RemapVoxels::~RemapVoxels()
{}

void RemapVoxels::okSlot()
{
  if(_minValue->text().toDouble() > _maxValue->text().toDouble())
    {
      QMessageBox::critical( this, "Input error", "Minimum value should be <= maximum value." );
      return;
    }

  accept();
}
