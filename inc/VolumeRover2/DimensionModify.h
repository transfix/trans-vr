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

/* $Id: DimensionModify.h 2159 2010-07-01 22:37:42Z transfix $ */

#ifndef __DIMENSIONMODIFY_H__
#define __DIMENSIONMODIFY_H__

#include <QDialog>
#include <VolMagick/Dimension.h>

namespace Ui
{
  class DimensionModify;
}

class DimensionModify : public QDialog
{
  Q_OBJECT

 public:
  DimensionModify(QWidget* parent = nullptr,
                  Qt::WindowFlags flags=Qt::WindowFlags());
  virtual ~DimensionModify();

  VolMagick::Dimension dimension() const;
  void dimension(const VolMagick::Dimension &dim);

 protected slots:
  virtual void okSlot();

 protected:
#if QT_VERSION < 0x040000
  DimensionModifyBase *_ui;
#else
  Ui::DimensionModify *_ui;
#endif
};

#endif
