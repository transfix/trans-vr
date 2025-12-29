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

#ifndef __BILATERALFILTERDIALOG_H__
#define __BILATERALFILTERDIALOG_H__

// #include <qglobal.h>

#include <QDialog>
#include <QString>

namespace Ui {
class BilateralFilterDialog;
}

#include "ui_BilateralFilterDialog.h"

class BilateralFilterDialog : public QDialog {
  Q_OBJECT

public:
  BilateralFilterDialog(QWidget *parent = nullptr,
                        Qt::WindowFlags flags = Qt::WindowFlags());

  virtual ~BilateralFilterDialog();

  void RunBilateralFilter();

public slots:

  void OutputFileSlot();

protected:
  Ui::BilateralFilterDialog *_ui;
};

#endif
