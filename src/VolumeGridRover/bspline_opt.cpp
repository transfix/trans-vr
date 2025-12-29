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

/* $Id: bspline_opt.cpp 4741 2011-10-21 21:22:06Z transfix $ */

#include <VolumeGridRover/bspline_opt.h>
#include <qbuttongroup.h>
#include <qcheckbox.h>
#include <qlabel.h>
#include <qlayout.h>
#include <qlineedit.h>
#include <qpushbutton.h>
#include <qradiobutton.h>
#include <qvalidator.h>

bspline_opt::bspline_opt(QWidget *parent, const char *name, bool modal) {
  setCaption("B-Spline Output Options");

  QGridLayout *base_layout = new QGridLayout(this, 5, 2, 11, 6);
  base_layout->setResizeMode(QLayout::Fixed);

  // left side labels
  QLabel *degree_label = new QLabel("Degree: ", this);
  QLabel *curslice_label =
      new QLabel("Save only current slice's contours: ", this);
  base_layout->addWidget(degree_label, 0, 0);
  base_layout->addWidget(curslice_label, 1, 0);

  // right side input
  degree = new QLineEdit(this);
  curslice = new QCheckBox(this);
  base_layout->addWidget(degree, 0, 1);
  base_layout->addWidget(curslice, 1, 1);

  degree->setText("3"); // default to a degree of 3
  degree->setValidator(new QIntValidator(this));

  // points and normals? or knots and control points.. you decide!
  spline_output_type =
      new QButtonGroup(1, Qt::Horizontal, "Spline Output", this);
  QRadioButton *kcp =
      new QRadioButton("Knots and Control Points", spline_output_type);
  kcp->setChecked(true);
  QRadioButton *pn =
      new QRadioButton("Points and Normals", spline_output_type);
  // pn->setChecked(true);
  base_layout->addMultiCellWidget(spline_output_type, 2, 2, 0, 1);
  connect(spline_output_type, SIGNAL(clicked(int)),
          SLOT(toggleControlPointOutputTypeFunctionality(int)));

  // control point output type
  control_type =
      new QButtonGroup(1, Qt::Horizontal, "Control Point Type", this);
  QRadioButton *fs = new QRadioButton(
      "Fit Spline and Generate Control Points", control_type);
  // fs->setChecked(true);
  QRadioButton *cp =
      new QRadioButton("Contour Points as Control Points", control_type);
  cp->setChecked(true);
  base_layout->addMultiCellWidget(control_type, 3, 3, 0, 1);

  // bottom buttons
  QPushButton *ok = new QPushButton("Ok", this);
  QPushButton *cancel = new QPushButton("Cancel", this);
  base_layout->addWidget(ok, 4, 1);
  base_layout->addWidget(cancel, 4, 0);

  connect(ok, SIGNAL(clicked()), SLOT(accept()));
  connect(cancel, SIGNAL(clicked()), SLOT(reject()));
}

bspline_opt::~bspline_opt() {}

void bspline_opt::toggleControlPointOutputTypeFunctionality(int id) {
  switch (id) {
  default:
  case 0:
    control_type->setEnabled(true);
    break;
  case 1:
    control_type->setEnabled(false);
    break;
  }
}
