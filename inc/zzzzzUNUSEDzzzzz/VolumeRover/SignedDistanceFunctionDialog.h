/*
  Copyright 2002-2003 The University of Texas at Austin
  
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

/* $Id: SignedDistanceFunctionDialog.h 3500 2011-01-25 16:51:16Z arand $ */

#ifndef __VOLUME__SIGNEDDISTANCEFUNCTIONDIALOG_H__
#define __VOLUME__SIGNEDDISTANCEFUNCTIONDIALOG_H__

#include "signeddistancefunctiondialogbase.Qt3.h"
//Added by qt3to4:
#include <QEvent>
#include <QCustomEvent>

class SignedDistanceFunctionDialog : public SignedDistanceFunctionDialogBase
{
Q_OBJECT
    
 public:
  SignedDistanceFunctionDialog( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, Qt::WFlags fl = 0 );
  ~SignedDistanceFunctionDialog();

 void grabSubVolBox();

 public slots:
  void setBoundingBox(double minx, double miny, double minz,
		      double maxx, double maxy, double maxz);
};

class SignedDistanceFunctionDialogGrabSubVolBoxEvent : public QCustomEvent
{
 public:
  SignedDistanceFunctionDialogGrabSubVolBoxEvent(SignedDistanceFunctionDialog *d)
    : QCustomEvent(QEvent::User+4000), dialog(d) {}

  SignedDistanceFunctionDialog *dialog;
};

#endif
