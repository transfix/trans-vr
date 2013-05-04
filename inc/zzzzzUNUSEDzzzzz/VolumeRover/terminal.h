/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
		 John Wiggins <prok@ices.utexas.edu>
		 Jose Rivera  <transfix@cs.utexas.edu>
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

#ifndef TERMINAL_H
#define TERMINAL_H

#include "terminalbase.Qt3.h"
//Added by qt3to4:
#include <QHideEvent>

class Terminal : public TerminalBase
{
    Q_OBJECT

 public:
  Terminal( QWidget* parent = 0, const char* name = 0, Qt::WFlags fl = 0 );
  ~Terminal();

 signals:
  void showToggle(bool show);

 protected:
  void hideEvent(QHideEvent *e);
};

#endif
