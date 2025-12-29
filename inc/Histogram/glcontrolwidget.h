/*
  Copyright 2011 The University of Texas at Austin

        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of TexMol.

  TexMol is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  TexMol is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/
#ifndef GLCONTROLWIDGET_H
#define GLCONTROLWIDGET_H

#include <QHideEvent>
#include <QMouseEvent>
#include <QShowEvent>
#include <QWheelEvent>
#include <qgl.h>

class GLControlWidget : public QGLWidget {
  Q_OBJECT

public:
  GLControlWidget(QWidget *parent, const char *name = 0, QGLWidget *share = 0,
                  Qt::WFlags f = 0);
  ~GLControlWidget() {}
  virtual void transform();

public slots:
  void setXRotation(double degrees);
  void setYRotation(double degrees);
  void setZRotation(double degrees);
  void setScale(double s);
  void setXTrans(double x);
  void setYTrans(double y);
  void setZTrans(double z);
  virtual void setRotationImpulse(double x, double y, double z);
  virtual void setTranslationImpulse(double x, double y, double z);
  void drawText();

protected:
  void setAnimationDelay(int ms);
  virtual void mousePressEvent(QMouseEvent *e);
  virtual void mouseReleaseEvent(QMouseEvent *e);
  virtual void mouseMoveEvent(QMouseEvent *);
  virtual void mouseDoubleClickEvent(QMouseEvent *);
  virtual void wheelEvent(QWheelEvent *);
  void showEvent(QShowEvent *);
  void hideEvent(QHideEvent *);
  GLfloat xRot, yRot, zRot;
  GLfloat xTrans, yTrans, zTrans;
  GLfloat scale;
  bool animation;

protected slots:
  virtual void animate();

private:
  bool wasAnimated;
  QPoint oldPos;
  QTimer *timer;
  int delay;
};

#endif
