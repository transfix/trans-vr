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

/* $Id: SignedDistanceFunctionDialog.cpp 3500 2011-01-25 16:51:16Z arand $ */

#include <qapplication.h>
#include <qlineedit.h>
#include <qvalidator.h>
#include <VolumeRover/SignedDistanceFunctionDialog.h>

SignedDistanceFunctionDialog::SignedDistanceFunctionDialog( QWidget* parent,  const char* name, bool modal, Qt::WFlags fl )
    : SignedDistanceFunctionDialogBase( parent, name, modal, fl )
{
  m_DimX->setValidator(new QIntValidator(this));
  m_DimY->setValidator(new QIntValidator(this));
  m_DimZ->setValidator(new QIntValidator(this));
  
  m_MinX->setValidator(new QDoubleValidator(this));
  m_MinY->setValidator(new QDoubleValidator(this));
  m_MinZ->setValidator(new QDoubleValidator(this));
  m_MaxX->setValidator(new QDoubleValidator(this));
  m_MaxY->setValidator(new QDoubleValidator(this));
  m_MaxZ->setValidator(new QDoubleValidator(this));
}

SignedDistanceFunctionDialog::~SignedDistanceFunctionDialog()
{
}

void SignedDistanceFunctionDialog::grabSubVolBox()
{
  QApplication::postEvent(qApp->mainWidget(),
			  new SignedDistanceFunctionDialogGrabSubVolBoxEvent(this));
}

void SignedDistanceFunctionDialog::setBoundingBox(double minx, double miny, double minz,
						  double maxx, double maxy, double maxz)
{
  m_MinX->setText(QString("%1").arg(minx));
  m_MinY->setText(QString("%1").arg(miny));
  m_MinZ->setText(QString("%1").arg(minz));
  m_MaxX->setText(QString("%1").arg(maxx));
  m_MaxY->setText(QString("%1").arg(maxy));
  m_MaxZ->setText(QString("%1").arg(maxz));
}
