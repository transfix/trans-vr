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

/* $Id: NewVolumeDialog.cpp 1527 2010-03-12 22:10:16Z transfix $ */

#include <qvalidator.h>
#include <qlineedit.h>
#include <qfiledialog.h>
#include <qfileinfo.h>
#include <qmessagebox.h>
#include <qbuttongroup.h>
#include <qcheckbox.h>
#include <qtabwidget.h>
#include <VolMagick/VolMagick.h>
#include <VolumeMaker/NewVolumeDialog.h>

NewVolumeDialog::NewVolumeDialog(QWidget* parent, const char* name, WFlags f)
  : NewVolumeDialogBase(parent,name,f)
{
  QIntValidator* intv = new QIntValidator(this);
  QDoubleValidator* doublev = new QDoubleValidator(this);

  _dimensionX->setValidator(intv);
  _dimensionY->setValidator(intv);
  _dimensionZ->setValidator(intv);

  _boundingBoxMinX->setValidator(doublev);
  _boundingBoxMinY->setValidator(doublev);
  _boundingBoxMinZ->setValidator(doublev);
  _boundingBoxMaxX->setValidator(doublev);
  _boundingBoxMaxY->setValidator(doublev);
  _boundingBoxMaxZ->setValidator(doublev);

  _extractSubVolumeMinIndexX->setValidator(intv);
  _extractSubVolumeMinIndexY->setValidator(intv);
  _extractSubVolumeMinIndexZ->setValidator(intv);
  _extractSubVolumeMaxIndexX->setValidator(intv);
  _extractSubVolumeMaxIndexY->setValidator(intv);
  _extractSubVolumeMaxIndexZ->setValidator(intv);

  _extractSubVolumeMinX->setValidator(doublev);
  _extractSubVolumeMinY->setValidator(doublev);
  _extractSubVolumeMinZ->setValidator(doublev);
  _extractSubVolumeMaxX->setValidator(doublev);
  _extractSubVolumeMaxY->setValidator(doublev);
  _extractSubVolumeMaxZ->setValidator(doublev);

  _extractSubVolumeBoundingBoxDimX->setValidator(intv);
  _extractSubVolumeBoundingBoxDimY->setValidator(intv);
  _extractSubVolumeBoundingBoxDimZ->setValidator(intv);

  _extractSubVolumeMethod->setEnabled(false);

  connect(_extractSubVolume,SIGNAL(toggled(bool)),SLOT(acquireVolumeInfo(bool)));
}

NewVolumeDialog::~NewVolumeDialog()
{

}

void NewVolumeDialog::okSlot()
{
  //error checking
  if(_newCopyGroup->selectedId() == 0)
    {
      if(_dimensionX->text().toInt() <= 0 ||
	 _dimensionY->text().toInt() <= 0 ||
	 _dimensionZ->text().toInt() <= 0)
	{
	  QMessageBox::critical( this, "Input error", "Dimension should be at least 1x1x1!" );
	  return;
	}
      
      if((_boundingBoxMaxX->text().toDouble() - _boundingBoxMinX->text().toDouble()) <= 0 ||
	 (_boundingBoxMaxY->text().toDouble() - _boundingBoxMinY->text().toDouble()) <= 0 ||
	 (_boundingBoxMaxZ->text().toDouble() - _boundingBoxMinZ->text().toDouble()) <= 0)
	{
	  QMessageBox::critical( this, "Input error", 
				 "Invalid bounding box!\n"
				 "Bounding box must have volume, and min < max");
	  return;
	}
    }
  else if(_newCopyGroup->selectedId() == 1)
    {
      if(_volumeCopyFilename->text().isEmpty())
	{
	  QMessageBox::critical( this, "Input error", "Please specify a filename for the volume to copy" );
	  return;
	}
    }

  if(_filename->text().isEmpty())
    {
      QMessageBox::critical( this, "Input error", "Please specify a filename for this new volume" );
      return;
    }

  while(QFileInfo(_filename->text()).exists() || _filename->text().isEmpty())
    {
      if(!_filename->text().isEmpty())
	{
	  int retval = QMessageBox::warning(this,
					    "Warning",
					    "File exists! Overwrite?",
					    QMessageBox::No,
					    QMessageBox::Yes);
	  if(retval == QMessageBox::No)
	    fileSlot();
	  else
	    break;
	}
      else if(_filename->text().isEmpty())
	{
	  QMessageBox::critical( this, "Input error", "Please specify a filename for this new volume" );
	  return;
	}
    }

  accept();
}

void NewVolumeDialog::fileSlot()
{
  _filename->setText(QFileDialog::getSaveFileName(QString::null,
						  "RawIV (*.rawiv);;"
						  "RawV (*.rawv);;"
						  "MRC (*.mrc);;"
						  "INR (*.inr);;"
						  "Spider (*.vol *.xmp *.spi);;"
						  "All Files (*)",
						  this));
}

void NewVolumeDialog::volumeCopyFilenameSlot()
{
  _volumeCopyFilename->setText(QFileDialog::getOpenFileName(QString::null,
							    "RawIV (*.rawiv);;"
							    "RawV (*.rawv);;"
							    "MRC (*.mrc);;"
							    "INR (*.inr);;"
							    "Spider (*.vol *.xmp *.spi);;"
							    "All Files (*)",
							    this));
}

void NewVolumeDialog::acquireVolumeInfo(bool doit)
{
  if(doit)
    {
      if(_volumeCopyFilename->text().isEmpty())
	return;
      
      VolMagick::VolumeFileInfo vfi(_volumeCopyFilename->text());
      _extractSubVolumeMinIndexX->setText("0");
      _extractSubVolumeMinIndexY->setText("0");
      _extractSubVolumeMinIndexZ->setText("0");
      _extractSubVolumeMaxIndexX->setText(QString("%1").arg(vfi.XDim()-1));
      _extractSubVolumeMaxIndexY->setText(QString("%1").arg(vfi.YDim()-1));
      _extractSubVolumeMaxIndexZ->setText(QString("%1").arg(vfi.ZDim()-1));
      _extractSubVolumeMinX->setText(QString("%1").arg(vfi.XMin()));
      _extractSubVolumeMinY->setText(QString("%1").arg(vfi.YMin()));
      _extractSubVolumeMinZ->setText(QString("%1").arg(vfi.ZMin()));
      _extractSubVolumeMaxX->setText(QString("%1").arg(vfi.XMax()));
      _extractSubVolumeMaxY->setText(QString("%1").arg(vfi.YMax()));
      _extractSubVolumeMaxZ->setText(QString("%1").arg(vfi.ZMax()));
      _extractSubVolumeBoundingBoxDimX->setText(QString("%1").arg(vfi.XDim()));
      _extractSubVolumeBoundingBoxDimY->setText(QString("%1").arg(vfi.YDim()));
      _extractSubVolumeBoundingBoxDimZ->setText(QString("%1").arg(vfi.ZDim()));
    }
}
