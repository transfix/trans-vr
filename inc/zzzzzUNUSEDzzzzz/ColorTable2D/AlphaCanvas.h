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

#ifndef ALPHACANVAS_H
#define ALPHACANVAS_H

#include <ColorTable2D/TableCanvas.h>
#include <qptrlist.h>

class AlphaCanvas : public TableCanvas
{
  Q_OBJECT
 public:

  AlphaCanvas( QWidget *parent, const char *name );
  ~AlphaCanvas();

 protected:

  class AlphaNode
  {
  public:
    unsigned char alpha;
    double m_X, m_Y; // 0 <= m_X <= 1, 0 <= m_Y <= 1
    
    AlphaNode(unsigned char a, double x, double y);
    ~AlphaNode();
  };

  //  void drawContents( QPainter *painter );
  QPtrList<AlphaNode> m_AlphaNodes;
};

#endif
