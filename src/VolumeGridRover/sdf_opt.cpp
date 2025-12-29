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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

/* $Id: sdf_opt.cpp 4741 2011-10-21 21:22:06Z transfix $ */

#include <VolumeGridRover/sdf_opt.h>
#include <qcheckbox.h>
#include <qcombobox.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qlineedit.h>
#include <qpushbutton.h>
#include <qvalidator.h>

sdf_opt::sdf_opt(QWidget *parent, const char *name, bool modal)
    : QDialog(parent) {
  setWindowTitle("2D SDF Calculation");

  QGridLayout *base_layout = new QGridLayout(this);
  base_layout->setSizeConstraint(QLayout::SetFixedSize);

  // left side labels
  QLabel *x_sample_res_label = new QLabel("X sample res: ", this);
  QLabel *y_sample_res_label = new QLabel("Y sample res: ", this);
  QLabel *sign_method_label = new QLabel("Sign calc method: ", this);
  QLabel *dist_method_label = new QLabel("Distance calc method: ", this);
  QLabel *sign_bitmap_only_label =
      new QLabel("Output only sign bitmap? ", this);
  base_layout->addWidget(x_sample_res_label, 0, 0);
  base_layout->addWidget(y_sample_res_label, 1, 0);
  base_layout->addWidget(sign_method_label, 2, 0);
  base_layout->addWidget(dist_method_label, 3, 0);
  base_layout->addWidget(sign_bitmap_only_label, 4, 0);

  // right side input
  x_sample_res = new QLineEdit(this);
  y_sample_res = new QLineEdit(this);
  x_sample_res->setText("128");
  x_sample_res->setValidator(new QIntValidator(this));
  y_sample_res->setText("128");
  y_sample_res->setValidator(new QIntValidator(this));
  sign_method = new QComboBox(this);
  sign_method->insertItem(0, "Angle Sum");
  sign_method->insertItem(1, "Count Edge Intersections");
  dist_method = new QComboBox(this);
  dist_method->insertItem(0, "Brute Force");
  dist_method->insertItem(1, "K Neighbor Search");
  dist_method->setCurrentIndex(1);
  sign_bitmap_only = new QCheckBox(this);
  base_layout->addWidget(x_sample_res, 0, 1);
  base_layout->addWidget(y_sample_res, 1, 1);
  base_layout->addWidget(sign_method, 2, 1);
  base_layout->addWidget(dist_method, 3, 1);
  base_layout->addWidget(sign_bitmap_only, 4, 1);

  // bottom buttons
  QPushButton *ok = new QPushButton("Run", this);
  QPushButton *cancel = new QPushButton("Cancel", this);
  base_layout->addWidget(ok, 5, 1);
  base_layout->addWidget(cancel, 5, 0);

  connect(ok, SIGNAL(clicked()), SLOT(accept()));
  connect(cancel, SIGNAL(clicked()), SLOT(reject()));
}

sdf_opt::~sdf_opt() {}
