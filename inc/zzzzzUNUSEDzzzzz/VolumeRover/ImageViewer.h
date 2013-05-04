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

// ImageViewer.h: interface for the ImageViewer class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_IMAGEVIEWER_H__40FB182F_5404_4CE7_9521_E84FAB389EE0__INCLUDED_)
#define AFX_IMAGEVIEWER_H__40FB182F_5404_4CE7_9521_E84FAB389EE0__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "imageviewerbase.Qt3.h"
//Added by qt3to4:
#include <QPixmap>

class QPixmap;

class ImageViewer : public ImageViewerBase  
{
	Q_OBJECT
public:
  ImageViewer( QWidget* parent = 0, const char* name = 0, Qt::WFlags fl = Qt::WType_TopLevel );
	virtual ~ImageViewer();

	void setPixmap(const QPixmap& pixmap);

	void saveAsSlot();

};

#endif // !defined(AFX_IMAGEVIEWER_H__40FB182F_5404_4CE7_9521_E84FAB389EE0__INCLUDED_)
