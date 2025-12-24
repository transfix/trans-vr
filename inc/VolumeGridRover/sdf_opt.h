/*
  Copyright 2005-2008 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeGridRover.

  VolumeGridRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeGridRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: sdf_opt.h 4741 2011-10-21 21:22:06Z transfix $ */

#ifndef __SDF_OPT_H__
#define __SDF_OPT_H__

#include <QDialog>

class QLineEdit;
class QComboBox;
class QCheckBox;

class sdf_opt : public QDialog
{
  Q_OBJECT

 public:
  sdf_opt(QWidget *parent = 0, const char *name = 0, bool modal = false);
  ~sdf_opt();

 public:
  QLineEdit *x_sample_res;
  QLineEdit *y_sample_res;
  QComboBox *sign_method;
  QComboBox *dist_method;
  QCheckBox *sign_bitmap_only;
};

#endif
