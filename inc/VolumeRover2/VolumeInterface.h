/*
  Copyright 2008 The University of Texas at Austin

        Authors: Jose Rivera <transfix@ices.utexas.edu>
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

/* $Id: VolumeInterface.h 3602 2011-02-12 00:02:44Z transfix $ */

#ifndef __VOLUMEINTERFACE_H__
#define __VOLUMEINTERFACE_H__

#include <VolMagick/VolMagick.h>
#include <VolumeRover2/DataWidget.h>

namespace Ui {
class VolumeInterface;
}

class VolumeInterface : public CVC_NAMESPACE::DataWidget {
  Q_OBJECT

public:
  VolumeInterface(
      const VolMagick::VolumeFileInfo &vfi = VolMagick::VolumeFileInfo(),
      QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags());
  virtual ~VolumeInterface();

  virtual void initialize(const boost::any &datum) {
    VolMagick::VolumeFileInfo vfi =
        boost::any_cast<VolMagick::VolumeFileInfo>(datum);
    setInterfaceInfo(vfi);
  }

  void setInterfaceInfo(const VolMagick::VolumeFileInfo &vfi,
                        bool announce = false);
  VolMagick::VolumeFileInfo volumeFileInfo() const { return _vfi; }

protected slots:
  void dimensionModifySlot();
  void boundingBoxModifySlot();

  void addTimestepSlot();
  void addVariableSlot();
  void deleteTimestepSlot();
  void deleteVariableSlot();
  void editVariableSlot();
  void importDataSlot();
  void remapSlot();

signals:
  void volumeModified(const VolMagick::VolumeFileInfo &vfi);

protected:
  void getSelectedVarTime(int &var, int &time);

  VolMagick::VolumeFileInfo _vfi;

  Ui::VolumeInterface *_ui;
};

#endif
