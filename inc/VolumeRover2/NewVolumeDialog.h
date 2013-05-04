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

/* $Id: NewVolumeDialog.h 2122 2010-06-28 02:06:38Z transfix $ */

#ifndef __NEWVOLUMEDIALOG_H__
#define __NEWVOLUMEDIALOG_H__

#include <qglobal.h>

#if QT_VERSION < 0x040000
#include <qdialog.h>
#else
#include <QDialog>
#endif

#if QT_VERSION < 0x040000
class NewVolumeDialogBase;
#else
namespace Ui
{
  class NewVolumeDialog;
}
#endif

#include <VolMagick/BoundingBox.h>
#include <VolMagick/Dimension.h>

#include <string>

class NewVolumeDialog : public QDialog
{
  Q_OBJECT

 public:
  NewVolumeDialog(QWidget* parent = 0,
#if QT_VERSION < 0x040000
                  const char* name = 0, WFlags f = WType_TopLevel
#else
                  Qt::WFlags flags=0
#endif
                  );
  virtual ~NewVolumeDialog();

  bool createNewVolume() const;

  VolMagick::Dimension dimension() const;
  VolMagick::BoundingBox boundingBox() const;
  VolMagick::VoxelType variableType() const;
  std::string variableName() const;
  std::string filename() const;
  std::string volumeCopyFilename() const;
  bool extractSubVolume() const;
  
  enum ExtractSubVolumeMethod { INDICES, BOUNDING_BOX };
  ExtractSubVolumeMethod extractSubVolumeMethod() const;

  //INDICES
  VolMagick::IndexBoundingBox extractIndexSubVolume() const;

  //BOUNDING_BOX
  VolMagick::BoundingBox extractSubVolumeBoundingBox() const;
  VolMagick::Dimension extractSubVolumeDimension() const;
  
 protected slots:
  void okSlot();
  void fileSlot();
  void volumeCopyFilenameSlot();

  void acquireVolumeInfo(bool);

 protected:
#if QT_VERSION < 0x040000
  NewVolumeDialogBase *_ui;
#else
  Ui::NewVolumeDialog *_ui;
#endif
};

#endif
