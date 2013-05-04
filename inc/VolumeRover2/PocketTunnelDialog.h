/*
  Copyright 2011 The University of Texas at Austin
  
	Authors: Alex Rand <arand@ices.utexas.edu>
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

#ifndef __POCKETTUNNELDIALOG_H__
#define __POCKETTUNNELDIALOG_H__

//#include <qglobal.h>

#include <QDialog>
#include <QString>

namespace Ui
{
  class PocketTunnelDialog;
}

#include "ui_PocketTunnelDialog.h"
      
class PocketTunnelDialog : public QDialog
{
  Q_OBJECT

 public:
  PocketTunnelDialog(QWidget *parent=0,Qt::WFlags flags=0);

  virtual ~PocketTunnelDialog();

  void RunPocketTunnel();

  
  public slots:


  protected:

  Ui::PocketTunnelDialog *_ui;

};

#endif
