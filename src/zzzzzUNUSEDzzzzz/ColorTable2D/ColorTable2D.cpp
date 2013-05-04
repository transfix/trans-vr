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
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <qframe.h>
#include <qlayout.h>
#include <ColorTable2D/ColorTable2D.h>
#include <ColorTable2D/AlphaCanvas.h>
#include <ColorTable2D/ColorCanvas.h>

ColorTable2D::ColorTable2D( QWidget *parent, const char *name )
  : QFrame( parent, name )
{
  setFrameStyle( QFrame::Panel | QFrame::Raised );

  QBoxLayout *layout = new QBoxLayout( this, QBoxLayout::Down );
  layout->setMargin( 3 );
  layout->setSpacing( 3 );

  m_AlphaCanvas = new AlphaCanvas( this, "m_AlphaCanvas" );
  m_ColorCanvas = new ColorCanvas( this, "m_ColorCanvas" );
  layout->addWidget( m_AlphaCanvas );
  layout->addWidget( m_ColorCanvas );
}

ColorTable2D::~ColorTable2D()
{}

QSize ColorTable2D::sizeHint() const
{
  return QSize(150, 150);
}

void ColorTable2D::GetTransferFunction(double *pMap, int size)
{
  /* just make compiler errors go away for now until this is implemented. */
  if(size) printf("%lf\n",*pMap);
}
