/*
  Copyright 2002-2003 The University of Texas at Austin
  
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

#include <qpainter.h>
#include <qrect.h>
#include <ColorTable2D/TableCanvas.h>

TableCanvas::TableCanvas( QWidget *parent, const char *name )
  : QFrame( parent, name ), m_Xmin(0), m_Xmax(1), m_Ymin(0),
    m_Ymax(1)
{
  setFrameStyle( QFrame::Panel | QFrame::Sunken );

  //setBackgroundMode( Qt::NoBackground );
}

TableCanvas::~TableCanvas() {}

QRect TableCanvas::getMyRect() const
{
  QRect rect = contentsRect();
  return QRect(rect.x()+10, rect.y()+5, rect.width()-20, rect.height()-10);
}

void TableCanvas::drawContents( QPainter *painter) 
{
  painter->setPen( Qt::blue );
  painter->drawText( contentsRect(), AlignCenter, "Nothing here to see :(" );
}
