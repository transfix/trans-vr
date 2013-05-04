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

/* $Id: DimensionModify.cpp 4741 2011-10-21 21:22:06Z transfix $ */

#include <qglobal.h>

#if QT_VERSION < 0x040000
#include <qvalidator.h>
#include <qlineedit.h>
#include <qmessagebox.h>
#include <qpushbutton.h>
#else
#include <QIntValidator>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#endif

#include <VolumeRover2/DimensionModify.h>

#if QT_VERSION < 0x040000
#include "dimensionmodifybase.Qt3.h"
#else
#include "ui_DimensionModify.h"
#endif

DimensionModify::DimensionModify(QWidget* parent,
#if QT_VERSION < 0x040000
                                 const char* name, WFlags f
#else
                                 Qt::WFlags flags
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
  _ui = new DimensionModifyBase(this);
#else
  _ui = new Ui::DimensionModify;
  _ui->setupUi(this);
#endif

  QIntValidator* intv = new QIntValidator(this);

  _ui->_dimensionX->setValidator(intv);
  _ui->_dimensionY->setValidator(intv);
  _ui->_dimensionZ->setValidator(intv);

  connect(_ui->_ok,SIGNAL(clicked()),SLOT(okSlot()));
}

DimensionModify::~DimensionModify()
{ delete _ui; }

VolMagick::Dimension DimensionModify::dimension() const
{
  return VolMagick::Dimension(_ui->_dimensionX->text().toInt(),
                              _ui->_dimensionY->text().toInt(),
                              _ui->_dimensionZ->text().toInt());
}

void DimensionModify::dimension(const VolMagick::Dimension& dim)
{
  _ui->_dimensionX->setText(QString("%1").arg(dim.XDim()));
  _ui->_dimensionY->setText(QString("%1").arg(dim.YDim()));
  _ui->_dimensionZ->setText(QString("%1").arg(dim.ZDim()));
}

void DimensionModify::okSlot()
{
  if(_ui->_dimensionX->text().toInt() <= 0 ||
     _ui->_dimensionY->text().toInt() <= 0 ||
     _ui->_dimensionZ->text().toInt() <= 0)
    {
      QMessageBox::critical( this, "Input error", "Dimension should be at least 1x1x1!" );
      return;
    }

  accept();
}
