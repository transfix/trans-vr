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

#ifdef USING_TILING

#ifndef __CONTOURSINTERFACE_H__
#define __CONTOURSINTERFACE_H__

#include <VolumeRover2/DataWidget.h>
#include <cvcraw_geometry/contours.h>
#include <cvcraw_geometry/cvcraw_geometry.h>
#include <cvcraw_geometry/io.h>

namespace Ui {
class ContoursInterface;
}

class ContoursInterface : public CVC_NAMESPACE::DataWidget {
  Q_OBJECT

public:
  ContoursInterface(
      const cvcraw_geometry::contours_t &geom = cvcraw_geometry::contours_t(),
      QWidget *parent = 0, Qt::WindowFlags flags = Qt::WindowFlags());
  virtual ~ContoursInterface();

  virtual void initialize(const boost::any &datum) {
    setInterfaceInfo(boost::any_cast<cvcraw_geometry::contours_t>(datum));
  }

  void setInterfaceInfo(const cvcraw_geometry::contours_t &geom);

signals:

public slots:
  void changeZ();

protected:
  Ui::ContoursInterface *_ui;

private:
  // cvcraw_geometry::contours_t* _contours;
  std::string _mapName;
};

#endif

#endif
