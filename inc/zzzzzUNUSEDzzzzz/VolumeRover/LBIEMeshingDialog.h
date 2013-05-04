/*
  Copyright 2002-2008 The University of Texas at Austin
  
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

/* $Id: LBIEMeshingDialog.h 3500 2011-01-25 16:51:16Z arand $ */

#ifndef __VOLUME__LBIEMESHINGDIALOG_H__
#define __VOLUME__LBIEMESHINGDIALOG_H__

#include "lbiemeshingdialogbase.Qt3.h"

class LBIEMeshingDialog : public LBIEMeshingDialogBase
{
Q_OBJECT

 public:
  LBIEMeshingDialog(QWidget *parent = 0, const char *name = 0, bool modal = FALSE, Qt::WFlags fl = 0);
  ~LBIEMeshingDialog();

 protected slots:
  void isovalueSelection(int);
  void meshTypeSelection(int);
  void extractionMethodSelection(int);
};

#endif
