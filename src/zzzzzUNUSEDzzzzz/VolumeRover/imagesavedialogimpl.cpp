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

#include <VolumeRover/imagesavedialogimpl.h>
//#include <qimage.h>
//#include <qcombobox>
#include <QComboBox>
//#include <qstrlist.h>
#include <QList>
#include <QImageWriter>


/* 
 *  Constructs a ImageSaveDialog which is a child of 'parent', with the 
 *  name 'name' and widget flags set to 'f' 
 *
 *  The dialog will by default be modeless, unless you set 'modal' to
 *  TRUE to construct a modal dialog.
 */
ImageSaveDialog::ImageSaveDialog( QWidget* parent,  const char* name, bool modal, Qt::WFlags fl )
    : ImageSaveDialogBase( parent, name, modal, fl )
{

  QList<QByteArray> formats = QImageWriter::supportedImageFormats();

    //QStrList formats = QImageIO::outputFormats();
	char *ptr=NULL;
	unsigned int counter=0;

	//qDebug("hey!! %d", formats.count());

	// find the JPEG entry in the list, remove it, and insert it at the
	// beginning
	while ((ptr = (char *)(formats.at(counter).data())) != NULL)
	{
		if (strcmp(ptr, "JPEG") == 0)
		{
			//qDebug("hey!");
		  QByteArray ptr = formats.takeAt(counter);
			//assert(ptr != NULL);
		  formats.insert(0, ptr);
		  break;
		}
		counter++;
	}

	// populate the image format combobox with image formats

	// skipping this for now... because of QString issues
	//imageFormatMenu->insertStrList(formats);
}

/*  
 *  Destroys the object and frees any allocated resources
 */
ImageSaveDialog::~ImageSaveDialog()
{
    // no need to delete child widgets, Qt does it all for us
}

