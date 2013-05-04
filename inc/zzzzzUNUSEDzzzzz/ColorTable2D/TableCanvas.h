/*
  Copyright 2002-2005 The University of Texas at Austin
  
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
  along with Volume Rover; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#ifndef TABLECANVAS_H
#define TABLECANVAS_H

#include <qframe.h>

class QRect;
class QPainter;

///\class TableCanvas TableCanvas.h
///\author Jose Rivera
///\brief This class does the bulk of the drawing and mouse handling for
/// the ColorTable2D widget. TODO: Write more!
class TableCanvas : public QFrame
{
  Q_OBJECT
 public:

  TableCanvas( QWidget *parent, const char *name );
  virtual ~TableCanvas();

 protected:
 
///\fn QRect getMyRect() const
///\brief Returns the rectangle that the widget draws inside of
///\return A QRect
  QRect getMyRect() const;

  virtual void drawContents( QPainter *painter );

  /* These variables define which part of the whole table canvas is rendered
     to the widget.  By changing the variables following the commented
     constraints below, we can zoom the users view of the table canvas for
     greater precision for manipulating the transfer function. */
  double m_Xmin, m_Xmax; // 0 <= m_Xmin <= 1, 0 <= m_Xmax <= 1, m_Xmin < m_Xmax
  double m_Ymin, m_Ymax; // 0 <= m_Ymin <= 1, 0 <= m_Ymax <= 1, m_Ymin < m_Ymax
};

#endif
