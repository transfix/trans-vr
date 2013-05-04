/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
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

// ImageViewer.cpp: implementation of the ImageViewer class.
//
//////////////////////////////////////////////////////////////////////

#include <VolumeRover/ImageViewer.h>
#include <qlabel.h>
//Added by qt3to4:
#include <QPixmap>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

ImageViewer::ImageViewer( QWidget* parent, const char* name, Qt::WFlags fl )
: ImageViewerBase( parent, name, fl )
{

}

ImageViewer::~ImageViewer()
{

}

void ImageViewer::setPixmap(const QPixmap& pixmap)
{
	m_PixmapLabel->setPixmap(pixmap);
	adjustSize();
}

void ImageViewer::saveAsSlot()
{

}

