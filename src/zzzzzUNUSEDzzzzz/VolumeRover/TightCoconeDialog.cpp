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

/* $Id: TightCoconeDialog.cpp 3500 2011-01-25 16:51:16Z arand $ */

#include <qlineedit.h>
#include <qvalidator.h>
#include <VolumeRover/TightCoconeDialog.h>

TightCoconeDialog::TightCoconeDialog( QWidget* parent,  const char* name, bool modal, Qt::WFlags fl )
    : TightCoconeDialogBase( parent, name, modal, fl )
{
  m_BigBallRatio->setValidator(new QDoubleValidator(this));
  m_ThetaIF->setValidator(new QDoubleValidator(this));
  m_ThetaFF->setValidator(new QDoubleValidator(this));
  m_FlatnessRatio->setValidator(new QDoubleValidator(this));
  m_CoconePhi->setValidator(new QDoubleValidator(this));
  m_FlatPhi->setValidator(new QDoubleValidator(this));
}

TightCoconeDialog::~TightCoconeDialog()
{
}
