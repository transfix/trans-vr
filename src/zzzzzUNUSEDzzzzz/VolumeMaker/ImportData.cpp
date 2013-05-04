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

/* $Id: ImportData.cpp 1527 2010-03-12 22:10:16Z transfix $ */

#include <qvalidator.h>
#include <qlineedit.h>
#include <qmessagebox.h>
#include <qfiledialog.h>
#include <qbuttongroup.h>
#include <VolMagick/VolMagick.h>
#include <VolumeMaker/ImportData.h>

ImportData::ImportData(QWidget* parent, const char* name, WFlags f)
  : ImportDataBase(parent,name,f)
{
  QIntValidator* intv = new QIntValidator(this);

  _offset->setValidator(intv);
  _dimensionX->setValidator(intv);
  _dimensionY->setValidator(intv);
  _dimensionZ->setValidator(intv);

  _variable->setValidator(intv);
  _timestep->setValidator(intv);
}

ImportData::~ImportData()
{}

void ImportData::importFileSlot()
{
  _importFile->setText(QFileDialog::getOpenFileName(QString::null,
						    "RawIV (*.rawiv);;"
						    "RawV (*.rawv);;"
						    "All Files (*)"
						    ,this));
}

void ImportData::okSlot()
{
  if(_importFile->text().isEmpty())
    {
      QMessageBox::critical( this, "Input error", "Specify a filename to import." );
      return;
    }

  if(_fileTypeGroup->selectedId() == 0)
    {
      if(_dimensionX->text().toInt() <= 0 ||
	 _dimensionY->text().toInt() <= 0 ||
	 _dimensionZ->text().toInt() <= 0)
	{
	  QMessageBox::critical( this, "Input error", "Dimension should be at least 1x1x1!" );
	  return;
	}
    }
  else if(_fileTypeGroup->selectedId() == 1)
    {
      VolMagick::VolumeFileInfo vfi(_importFile->text());

      if(_variable->text().toInt() >= int(vfi.numVariables()))
	{
	  QMessageBox::critical( this, "Input error", "Variable index greater than number of variables" );
	  return;
	}

      if(_timestep->text().toInt() >= int(vfi.numTimesteps()))
	{
	  QMessageBox::critical( this, "Input error", "Timestep index greater than number of timesteps" );
	  return;
	}
    }

  accept();
}
