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

/* $Id: LBIEMeshingDialog.cpp 3500 2011-01-25 16:51:16Z arand $ */

#include <VolumeRover/LBIEMeshingDialog.h>

#include <qvalidator.h>
#include <q3buttongroup.h>
#include <qlineedit.h>
#include <qcombobox.h>
#include <qcheckbox.h>

#include <LBIE/LBIE_Mesher.h>

LBIEMeshingDialog::LBIEMeshingDialog(QWidget *parent, const char *name, bool modal, Qt::WFlags fl)
  : LBIEMeshingDialogBase(parent,name,modal,fl)
{
  m_OuterIsoValue->setValidator(new QDoubleValidator(this));
  m_InnerIsoValue->setValidator(new QDoubleValidator(this));
  m_ErrorTolerance->setValidator(new QDoubleValidator(this));
  m_InnerErrorTolerance->setValidator(new QDoubleValidator(this));
  m_Iterations->setValidator(new QIntValidator(this));

  connect(m_WhichIsovaluesGroup,
	  SIGNAL(clicked(int)),
	  SLOT(isovalueSelection(int)));
  connect(m_MeshType,
	  SIGNAL(activated(int)),
	  SLOT(meshTypeSelection(int)));
  connect(m_MeshExtractionMethod,
	  SIGNAL(activated(int)),
	  SLOT(extractionMethodSelection(int)));

  m_OuterIsoValue->setText(QString("%1").arg(-1.0*LBIE::DEFAULT_IVAL));
  m_InnerIsoValue->setText(QString("%1").arg(-1.0*LBIE::DEFAULT_IVAL_IN));
  m_ErrorTolerance->setText(QString("%1").arg(LBIE::DEFAULT_ERR));
  m_InnerErrorTolerance->setText(QString("%1").arg(LBIE::DEFAULT_ERR_IN));
  m_Iterations->setText(QString("%1").arg(1));
}

LBIEMeshingDialog::~LBIEMeshingDialog() {}

void LBIEMeshingDialog::isovalueSelection(int sel)
{
  switch(sel)
    {
    case 0:
      m_IsovalueGroup->setEnabled(false);
      break;
    case 1:
      m_IsovalueGroup->setEnabled(true);
      break;
    default: break;
    }
}

void LBIEMeshingDialog::meshTypeSelection(int sel)
{
  switch(sel)
    {
    case 0:
    case 1:
    case 2:
    case 3:
      m_DualContouring->setEnabled(true);
      break;
    case 4:
    case 5:
      m_DualContouring->setChecked(true);
      m_DualContouring->setEnabled(false);
      break;
    }
}

void LBIEMeshingDialog::extractionMethodSelection(int sel)
{
  switch(sel)
    {
    case 0: //only enabled for duallib
      m_ErrorTolerance->setEnabled(true);
      m_InnerErrorTolerance->setEnabled(true);
      m_MeshType->setEnabled(true);
      m_NormalType->setEnabled(true);
      m_DualContouring->setEnabled(true);
      break;
    default:
      m_ErrorTolerance->setEnabled(false);
      m_InnerErrorTolerance->setEnabled(false);
      m_MeshType->setEnabled(false);
      m_MeshType->setCurrentItem(0);
      m_NormalType->setEnabled(false);
      m_DualContouring->setEnabled(false);
      m_DualContouring->setChecked(false);
      break;
    }
}
