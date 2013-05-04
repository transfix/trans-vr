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

/* $Id: BoundingBoxModify.h 2015 2010-06-21 16:28:56Z transfix $ */

#ifndef __BOUNDINGBOXMODIFY_H__
#define __BOUNDINGBOXMODIFY_H__

#if QT_VERSION < 0x040000
#include <qdialog.h>
#else
#include <QDialog>
#endif

#if QT_VERSION < 0x040000
class BoundingBoxModifyBase;
#else
namespace Ui
{
  class BoundingBoxModify;
}
#endif

#include <VolMagick/BoundingBox.h>

class BoundingBoxModify : public QDialog
{
  Q_OBJECT

 public:
  BoundingBoxModify(QWidget* parent = 0, 
#if QT_VERSION < 0x040000
                    const char* name = 0, WFlags f = WType_TopLevel
#else
                    Qt::WFlags flags=0
#endif
                    );
  virtual ~BoundingBoxModify();

  VolMagick::BoundingBox boundingBox() const;
  void boundingBox(const VolMagick::BoundingBox& bbox);

  double centerPointX() const;
  double centerPointY() const;
  double centerPointZ() const;

  void centerPoint(double,double,double);

  bool usingCenterPoint() const;

 protected slots:
  void okSlot();

 protected:

#if QT_VERSION < 0x040000
  BoundingBoxModifyBase *_ui;
#else
  Ui::BoundingBoxModify *_ui;
#endif
};

#endif
