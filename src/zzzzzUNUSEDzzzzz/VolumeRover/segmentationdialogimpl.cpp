/*
  Copyright 2002-2005 The University of Texas at Austin
  
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
  along with Volume Rover; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <VolumeRover/segmentationdialogimpl.h>
#include <qlineedit.h>
#include <q3filedialog.h>
#include <q3groupbox.h>
#include <qvalidator.h>

/* 
 *  Constructs a SegmentationDialog which is a child of 'parent', with the 
 *  name 'name' and widget flags set to 'f' 
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
SegmentationDialog::SegmentationDialog( QWidget* parent,  const char* name, bool modal, Qt::WFlags fl )
    : SegmentationDialogBase( parent, name, modal, fl )
{
	m_X0EditType0->setValidator(new QIntValidator(this));
	m_Y0EditType0->setValidator(new QIntValidator(this));
	m_Z0EditType0->setValidator(new QIntValidator(this));
	m_TLowEditType0->setValidator(new QDoubleValidator(0.0,255.0,5,this));
	m_TLowEditType1->setValidator(new QDoubleValidator(0.0,255.0,5,this));
	m_X0EditType1->setValidator(new QIntValidator(this));
	m_Y0EditType1->setValidator(new QIntValidator(this));
	m_Z0EditType1->setValidator(new QIntValidator(this));
	m_X1EditType1->setValidator(new QIntValidator(this));
	m_Y1EditType1->setValidator(new QIntValidator(this));
	m_Z1EditType1->setValidator(new QIntValidator(this));
	m_TLowEditType2->setValidator(new QDoubleValidator(0.0,255.0,5,this));
	m_SmallRadiusEditType2->setValidator(new QDoubleValidator(this));
	m_LargeRadiusEditType2->setValidator(new QDoubleValidator(this));
	m_TLowEditType3->setValidator(new QDoubleValidator(0.0,255.0,5,this));
	m_3FoldEditType3->setValidator(new QIntValidator(this));
	m_5FoldEditType3->setValidator(new QIntValidator(this));
	m_6FoldEditType3->setValidator(new QIntValidator(this));
	m_SmallRadiusEditType3->setValidator(new QDoubleValidator(this));
	m_LargeRadiusEditType3->setValidator(new QDoubleValidator(this));
	m_FoldNumEdit->setValidator(new QIntValidator(this));
	m_HNumEdit->setValidator(new QIntValidator(this));
	m_KNumEdit->setValidator(new QIntValidator(this));
	m_3FoldEdit->setValidator(new QIntValidator(this));
	m_5FoldEdit->setValidator(new QIntValidator(this));
	m_6FoldEdit->setValidator(new QIntValidator(this));
	m_InitRadiusEdit->setValidator(new QIntValidator(this));
	m_HelixWidth->setValidator(new QDoubleValidator(this));
	m_MinHelixWidthRatio->setValidator(new QDoubleValidator(this));
	m_MaxHelixWidthRatio->setValidator(new QDoubleValidator(this));
	m_MinHelixLength->setValidator(new QDoubleValidator(this));
	m_SheetWidth->setValidator(new QDoubleValidator(this));
	m_MinSheetWidthRatio->setValidator(new QDoubleValidator(this));
	m_MaxSheetWidthRatio->setValidator(new QDoubleValidator(this));
	m_SheetExtend->setValidator(new QDoubleValidator(this));
	m_Threshold->setValidator(new QDoubleValidator(this));

}

/*  
 *  Destroys the object and frees any allocated resources
 */
SegmentationDialog::~SegmentationDialog()
{
    // no need to delete child widgets, Qt does it all for us
}

void SegmentationDialog::changeExecutionLocationSlot(int loc)
{
	switch(loc)
	{
	case 0:
	m_RemoteSegmentationGroup->setEnabled(false);
	break;
	case 1:
	m_RemoteSegmentationGroup->setEnabled(true);
	break;
	}
}


