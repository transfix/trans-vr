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

#ifndef VOLUMEGRIDROVERMAINWINDOW_H
#define VOLUMEGRIDROVERMAINWINDOW_H

#include <QGridLayout>
#include <QObject>
#include <QWidget>
#include <VolMagick/VolMagick.h>

namespace CVC {
class ColorTable;
}

class VolumeGridRover;
class QBoxLayout;
class QToolBar;
// class MappedVolumeFile;

class VolumeGridRoverMainWindow : public QWidget {
  Q_OBJECT

public:
  VolumeGridRoverMainWindow(QWidget *parent = nullptr,
                            Qt::WindowFlags fl = Qt::WindowFlags());
  ~VolumeGridRoverMainWindow();

  void fileOpen();

public slots:
  void functionChangedSlot();

private:
  QToolBar *m_ColorToolbar;
  CVC::ColorTable *m_ColorTable;

  QGridLayout *m_VolumeGridRoverLayout;
  VolumeGridRover *m_VolumeGridRover;
  VolMagick::VolumeFileInfo m_VolumeFileInfo;
  // MappedVolumeFile *m_MappedVolumeFile;
};

#endif
