/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
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

#include <VolumeRover/optionsdialogimpl.h>
#include <qlineedit.h>
#include <q3filedialog.h>
#include <qpushbutton.h>
#include <qcolordialog.h>

/* 
 *  Constructs a OptionsDialog which is a child of 'parent', with the 
 *  name 'name' and widget flags set to 'f' 
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
OptionsDialog::OptionsDialog( QWidget* parent,  const char* name, bool modal, Qt::WFlags fl )
    : OptionsDialogBase( parent, name, modal, fl )
{
}

/*  
 *  Destroys the object and frees any allocated resources
 */
OptionsDialog::~OptionsDialog()
{
    // no need to delete child widgets, Qt does it all for us
}

void OptionsDialog::browseSlot()
{
	QDir dir(m_CacheDir->text());
	m_CacheDir->setText(presentDirDialog(dir).absPath());
}

void OptionsDialog::colorSlot()
{
	QColor color = QColorDialog::getColor( getColor() );
	if (color.isValid()) {
		setColor(color);
	}
}

QDir OptionsDialog::presentDirDialog(QDir defaultDir)
{
	// query the user
	QString s = Q3FileDialog::getExistingDirectory(
			defaultDir.absPath(),
			this, "Please choose the location of the cache directory.",
			QString("Please choose the location of the cache directory"), TRUE );
	if ( !(s.isNull()) ) {
		return QDir(s);
	}
	else {
		return defaultDir;
	}
}

void OptionsDialog::setColor(const QColor& backgroundColor)
{
	m_ColorButton->setPaletteBackgroundColor(backgroundColor);
}

const QColor& OptionsDialog::getColor()
{
	return m_ColorButton->paletteBackgroundColor();
}

