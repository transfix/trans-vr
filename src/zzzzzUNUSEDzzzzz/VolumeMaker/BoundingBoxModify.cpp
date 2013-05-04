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

/* $Id: BoundingBoxModify.cpp 1527 2010-03-12 22:10:16Z transfix $ */

#include <qvalidator.h>
#include <qlineedit.h>
#include <qmessagebox.h>
#include <qcheckbox.h>
#include <qgroupbox.h>
#include <VolumeMaker/BoundingBoxModify.h>

BoundingBoxModify::BoundingBoxModify(QWidget* parent, const char* name, WFlags f)
  : BoundingBoxModifyBase(parent,name,f)
{
  QDoubleValidator* doublev = new QDoubleValidator(this);

  _boundingBoxMinX->setValidator(doublev);
  _boundingBoxMinY->setValidator(doublev);
  _boundingBoxMinZ->setValidator(doublev);
  _boundingBoxMaxX->setValidator(doublev);
  _boundingBoxMaxY->setValidator(doublev);
  _boundingBoxMaxZ->setValidator(doublev);
  _centerPointX->setValidator(doublev);
  _centerPointY->setValidator(doublev);
  _centerPointZ->setValidator(doublev);

  connect(_useCenterPoint,
	  SIGNAL(toggled(bool)),
	  _boundingBoxGroup,
	  SLOT(setDisabled(bool)));
}

BoundingBoxModify::~BoundingBoxModify()
{}

void BoundingBoxModify::okSlot()
{
  if((_boundingBoxMaxX->text().toDouble() - _boundingBoxMinX->text().toDouble()) <= 0 ||
     (_boundingBoxMaxY->text().toDouble() - _boundingBoxMinY->text().toDouble()) <= 0 ||
     (_boundingBoxMaxZ->text().toDouble() - _boundingBoxMinZ->text().toDouble()) <= 0)
    {
      QMessageBox::critical( this, "Input error", 
			     "Invalid bounding box!\n"
			     "Bounding box must have volume, and min < max");
      return;
    }

  accept();
}
