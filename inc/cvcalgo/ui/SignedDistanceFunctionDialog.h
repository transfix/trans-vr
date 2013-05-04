/*
  Copyright 2008-2010 The University of Texas at Austin
  
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

/* $Id: SignedDistanceFunctionDialog.h 2273 2010-07-09 23:34:41Z transfix $ */

#ifndef __CVCALGO_UI_SIGNEDDISTANCEFUNCTIONDIALOG_H__
#define __CVCALGO_UI_SIGNEDDISTANCEFUNCTIONDIALOG_H__

#include <qglobal.h>

#if QT_VERSION < 0x040000
#include <qdialog.h>
#else
#include <QDialog>
#endif

#include <cvcalgo/cvcalgo.h>
#include <VolMagick/Dimension.h>

#if QT_VERSION < 0x040000
class SignedDistanceFunctionDialogBase;
#else
namespace Ui
{
  class SignedDistanceFunctionDialog;
}
#endif

class SignedDistanceFunctionDialog : public QDialog
{
  Q_OBJECT

 public:
  SignedDistanceFunctionDialog(QWidget* parent = 0,
#if QT_VERSION < 0x040000
                  const char* name = 0, WFlags f = WType_TopLevel
#else
                  Qt::WFlags flags=0
#endif
                  );
  virtual ~SignedDistanceFunctionDialog();

  cvcalgo::SDFMethod method() const;
  void method(cvcalgo::SDFMethod method);
  
  VolMagick::Dimension dimension() const;
  void dimension(const VolMagick::Dimension &dim);

  VolMagick::BoundingBox boundingBox() const;
  void boundingBox(const VolMagick::BoundingBox& bbox);

  bool usingBoundingBox() const;
  void usingBoundingBox(bool flag);

 signals:
  void getSubVolumeBoxButtonClicked();

 protected slots:
  virtual void okSlot();
  void enableBoundingBoxWidgets(bool);

 protected:
#if QT_VERSION < 0x040000
  SignedDistanceFunctionDialogBase *_ui;
#else
  Ui::SignedDistanceFunctionDialog *_ui;
#endif
};

#endif
