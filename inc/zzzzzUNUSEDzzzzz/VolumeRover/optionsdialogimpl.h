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

#ifndef OPTIONSDIALOG_H
#define OPTIONSDIALOG_H
#include "optionsdialog.Qt3.h"
#include <qdir.h>

class OptionsDialog : public OptionsDialogBase
{ 
    Q_OBJECT

public:
    OptionsDialog( QWidget* parent = 0, const char* name = 0, bool modal = FALSE, Qt::WFlags fl = 0 );
    ~OptionsDialog();

	virtual void browseSlot();
    virtual void colorSlot();
	QDir presentDirDialog(QDir defaultDir);

	void setColor(const QColor& backgroundColor);
	const QColor& getColor();

};

#endif // OPTIONSDIALOG_H
