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

/* $Id: BoundingBoxModify.cpp 4741 2011-10-21 21:22:06Z transfix $ */

#include <qglobal.h>

#if QT_VERSION < 0x040000
#include <qvalidator.h>
#include <qlineedit.h>
#include <qmessagebox.h>
#include <qcheckbox.h>
#include <qgroupbox.h>
#include <qpushbutton.h>
#else
#include <QDoubleValidator>
#include <QLineEdit>
#include <QMessageBox>
#include <QCheckBox>
#include <QGroupBox>
#include <QPushButton>
#endif

#include <VolumeRover2/BoundingBoxModify.h>

#if QT_VERSION < 0x040000
#include "boundingboxmodifybase.Qt3.h"
#else
#include "ui_BoundingBoxModify.h"
#endif

BoundingBoxModify::BoundingBoxModify(QWidget* parent, 
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
  _ui = new BoundingBoxModifyBase(this);
#else
  _ui = new Ui::BoundingBoxModify;
  _ui->setupUi(this);
#endif

  connect(_ui->_cancel, SIGNAL(clicked()), SLOT(reject()));
  connect(_ui->_ok, SIGNAL(clicked()), SLOT(okSlot()));

  QDoubleValidator* doublev = new QDoubleValidator(this);

  _ui->_boundingBoxMinX->setValidator(doublev);
  _ui->_boundingBoxMinY->setValidator(doublev);
  _ui->_boundingBoxMinZ->setValidator(doublev);
  _ui->_boundingBoxMaxX->setValidator(doublev);
  _ui->_boundingBoxMaxY->setValidator(doublev);
  _ui->_boundingBoxMaxZ->setValidator(doublev);
  _ui->_centerPointX->setValidator(doublev);
  _ui->_centerPointY->setValidator(doublev);
  _ui->_centerPointZ->setValidator(doublev);

  boundingBox(VolMagick::BoundingBox());
  centerPoint(0.0,0.0,0.0);
}

BoundingBoxModify::~BoundingBoxModify() { delete _ui; }

VolMagick::BoundingBox BoundingBoxModify::boundingBox() const
{
  return VolMagick::BoundingBox(_ui->_boundingBoxMinX->text().toDouble(),
                                _ui->_boundingBoxMinY->text().toDouble(),
                                _ui->_boundingBoxMinZ->text().toDouble(),
                                _ui->_boundingBoxMaxX->text().toDouble(),
                                _ui->_boundingBoxMaxY->text().toDouble(),
                                _ui->_boundingBoxMaxZ->text().toDouble());
}

void BoundingBoxModify::boundingBox(const VolMagick::BoundingBox& bbox)
{
  _ui->_boundingBoxMinX->setText(QString("%1").arg(bbox.XMin()));
  _ui->_boundingBoxMinY->setText(QString("%1").arg(bbox.YMin()));
  _ui->_boundingBoxMinZ->setText(QString("%1").arg(bbox.ZMin()));
  _ui->_boundingBoxMaxX->setText(QString("%1").arg(bbox.XMax()));
  _ui->_boundingBoxMaxY->setText(QString("%1").arg(bbox.YMax()));
  _ui->_boundingBoxMaxZ->setText(QString("%1").arg(bbox.ZMax()));
}

double BoundingBoxModify::centerPointX() const
{
  return _ui->_centerPointX->text().toDouble();
}

double BoundingBoxModify::centerPointY() const
{
  return _ui->_centerPointY->text().toDouble();
}

double BoundingBoxModify::centerPointZ() const
{
  return _ui->_centerPointZ->text().toDouble();
}

void BoundingBoxModify::centerPoint(double x, double y, double z)
{
  _ui->_centerPointX->setText(QString("%1").arg(x));
  _ui->_centerPointY->setText(QString("%1").arg(y));
  _ui->_centerPointZ->setText(QString("%1").arg(z));
}

bool BoundingBoxModify::usingCenterPoint() const
{
  return _ui->_useCenterPoint->isChecked();
}

void BoundingBoxModify::okSlot()
{
  if((_ui->_boundingBoxMaxX->text().toDouble() - _ui->_boundingBoxMinX->text().toDouble()) <= 0 ||
     (_ui->_boundingBoxMaxY->text().toDouble() - _ui->_boundingBoxMinY->text().toDouble()) <= 0 ||
     (_ui->_boundingBoxMaxZ->text().toDouble() - _ui->_boundingBoxMinZ->text().toDouble()) <= 0)
    {
      QMessageBox::critical( this, "Input error", 
			     "Invalid bounding box!\n"
			     "Bounding box must have volume, and min < max");
      return;
    }

  accept();
}
