/*
  Copyright 2008-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeRover.

  VolumeRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: RemapVoxels.cpp 4741 2011-10-21 21:22:06Z transfix $ */

#include <qglobal.h>

#if QT_VERSION < 0x040000
#include <qvalidator.h>
#include <qlineedit.h>
#include <qmessagebox.h>
#include <qpushbutton.h>
#else
#include <QDoubleValidator>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#endif

#include <VolumeRover2/RemapVoxels.h>

#if QT_VERSION < 0x040000
#include "remapvoxelsbase.Qt3.h"
#else
#include "ui_RemapVoxels.h"
#endif

RemapVoxels::RemapVoxels(QWidget* parent,
#if QT_VERSION < 0x040000
                                 const char* name, WFlags f
#else
                                 Qt::WindowFlags flags
#endif
                                 )
  : QDialog(parent,
#if QT_VERSION < 0x040000
            name,false,f
#else
            flags
#endif
            ),
    _ui(NULL)
{
#if QT_VERSION < 0x040000
  _ui = new RemapVoxelsBase(this);
#else
  _ui = new Ui::RemapVoxels;
  _ui->setupUi(this);
#endif

  QDoubleValidator* doublev = new QDoubleValidator(this);
  _ui->_minValue->setValidator(doublev);
  _ui->_maxValue->setValidator(doublev);

  connect(_ui->_ok,SIGNAL(clicked()),SLOT(okSlot()));
}

RemapVoxels::~RemapVoxels()
{ delete _ui; }

double RemapVoxels::minValue() const
{
  return _ui->_minValue->text().toDouble();
}

void RemapVoxels::minValue(double val)
{
  _ui->_minValue->setText(QString("%1").arg(val));
}

double RemapVoxels::maxValue() const
{
  return _ui->_maxValue->text().toDouble();
}

void RemapVoxels::maxValue(double val)
{
  _ui->_maxValue->setText(QString("%1").arg(val));
}

void RemapVoxels::okSlot()
{
  if(_ui->_minValue->text().toDouble() > _ui->_maxValue->text().toDouble())
    {
      QMessageBox::critical( this, "Input error", "Minimum value should be <= maximum value." );
      return;
    }

  accept();
}
