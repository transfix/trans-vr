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

#ifndef COLORCANVAS_H
#define COLORCANVAS_H

#include <ColorTable2D/TableCanvas.h>
#include <qptrlist.h>

class QColor;

class ColorCanvas : public TableCanvas
{
  Q_OBJECT
 public:
  ColorCanvas( QWidget *parent, const char *name );
  ~ColorCanvas();

 protected:

  class ColorNode
  {
  public:
    QColor color;
    double m_X, m_Y; // 0 <= m_X <= 1, 0 <= m_Y <= 1
    
    ColorNode(int r, int g, int b, double x, double y);
    ~ColorNode();
  };

  //void drawContents( QPainter *painter );

  // A list that holds all color nodes of the table.
  QPtrList<ColorNode> m_ColorNodes;
};

#endif
