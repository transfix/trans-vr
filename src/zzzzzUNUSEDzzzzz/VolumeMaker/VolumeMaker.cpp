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

/* $Id: VolumeMaker.cpp 1527 2010-03-12 22:10:16Z transfix $ */

#include <qlineedit.h>
#include <qwidgetstack.h>
#include <qtabwidget.h>
#include <qlayout.h>
#include <qfileinfo.h>
#include <qfiledialog.h>
#include <qbuttongroup.h>
#include <qmessagebox.h>
#include <qcombobox.h>
#include <qcheckbox.h>
#include <qlistview.h>
#include <VolumeMaker/VolumeMaker.h>
#include <VolumeMaker/NewVolumeDialog.h>
#include <VolumeMaker/VolumeInterface.h>

VolumeMaker::VolumeMaker( QWidget* parent, const char* name, WFlags f )
  : VolumeMakerBase(parent,name,f), 
    _volumeStack(NULL),
    _volumeStackList(NULL),
    _layout(new QGridLayout(_mainFrame,1,2))
{
  _volumeStackList = new QListView(_mainFrame);
  _layout->addWidget(_volumeStackList,0,0);
  _volumeStackList->addColumn("File Name");
  _volumeStackList->addColumn("Full Path");
  _volumeStackList->setSorting(-1);
  _volumeStackList->setSelectionMode(QListView::Extended);
  _volumeStackList->show();
  connect(_volumeStackList,
	  SIGNAL(selectionChanged()),
	  SLOT(raiseSelected()));
  
  _volumeStack = new QWidgetStack(_mainFrame);
  _layout->addWidget(_volumeStack,0,1);
  _volumeStack->show();

  _layout->setColStretch(0,3);
  _layout->setColStretch(1,2);
}

VolumeMaker::~VolumeMaker() {}

void VolumeMaker::newVolumeSlot()
{
  //qDebug("newvolumeslot");

  NewVolumeDialog nvd;

  if(nvd.exec() == QDialog::Accepted)
    {
      try
	{
	  if(nvd._newCopyGroup->selectedId() == 0) //new volume
	    {
	      //output the volume
	      VolMagick::Volume vol(VolMagick::Dimension(nvd._dimensionX->text().toInt(),
							 nvd._dimensionY->text().toInt(),
							 nvd._dimensionZ->text().toInt()),
				    VolMagick::VoxelType(nvd._variableType->currentItem()),
				    VolMagick::BoundingBox(nvd._boundingBoxMinX->text().toDouble(),
							   nvd._boundingBoxMinY->text().toDouble(),
							   nvd._boundingBoxMinZ->text().toDouble(),
							   nvd._boundingBoxMaxX->text().toDouble(),
							   nvd._boundingBoxMaxY->text().toDouble(),
							   nvd._boundingBoxMaxZ->text().toDouble()));
	      vol.desc(nvd._variableName->text());
	      VolMagick::createVolumeFile(nvd._filename->text(),
					  vol.boundingBox(),
					  vol.dimension(),
					  std::vector<VolMagick::VoxelType>(1,vol.voxelType()));
	      VolMagick::writeVolumeFile(vol,nvd._filename->text());
	  
	      //now read it's info
	      VolMagick::VolumeFileInfo vfi(nvd._filename->text());
	  
	      addVolumeToStack(vfi);
	    }
	  else //copy volume
	    {
	      VolMagick::VolumeFileInfo vfi_copy(nvd._volumeCopyFilename->text().ascii());

	      VolMagick::BoundingBox bb = vfi_copy.boundingBox();
	      
	      if(nvd._extractSubVolume->isChecked()) //extract sub volume
		{
		  switch(nvd._extractSubVolumeMethod->currentPageIndex())
		    {
		    case 0: //using image indices
		      {
			VolMagick::uint64 min_index[3] =
			  {
			    nvd._extractSubVolumeMinIndexX->text().toInt(),
			    nvd._extractSubVolumeMinIndexY->text().toInt(),
			    nvd._extractSubVolumeMinIndexZ->text().toInt(),
			  };
			
			VolMagick::uint64 max_index[3] =
			  {
			    nvd._extractSubVolumeMaxIndexX->text().toInt(),
			    nvd._extractSubVolumeMaxIndexY->text().toInt(),
			    nvd._extractSubVolumeMaxIndexZ->text().toInt(),
			  };
			
			VolMagick::Dimension d(max_index[0]-min_index[0]+1,
					       max_index[1]-min_index[1]+1,
					       max_index[2]-min_index[2]+1);
			VolMagick::BoundingBox bb(vfi_copy.XMin()+vfi_copy.XSpan()*min_index[0],
						  vfi_copy.YMin()+vfi_copy.YSpan()*min_index[1],
						  vfi_copy.ZMin()+vfi_copy.ZSpan()*min_index[2],
						  vfi_copy.XMin()+vfi_copy.XSpan()*max_index[0],
						  vfi_copy.YMin()+vfi_copy.YSpan()*max_index[1],
						  vfi_copy.ZMin()+vfi_copy.ZSpan()*max_index[2]);
			VolMagick::createVolumeFile(nvd._filename->text(),
						    bb,
						    d,
						    vfi_copy.voxelTypes(),
						    vfi_copy.numVariables(),
						    vfi_copy.numTimesteps(),
						    vfi_copy.TMin(),
						    vfi_copy.TMax());

			for(unsigned int var = 0; var < vfi_copy.numVariables(); var++)
			  for(unsigned int time = 0; time < vfi_copy.numTimesteps(); time++)
			    {
			      VolMagick::Volume vol;
			      VolMagick::readVolumeFile(vol,vfi_copy.filename(),var,time);
			      vol.sub(min_index[0],min_index[1],min_index[2],d);
			      VolMagick::writeVolumeFile(vol,nvd._filename->text(),var,time);	
			    }
		      }
		      break;
		    case 1: //using bounding box
		      {
			VolMagick::BoundingBox bb(nvd._extractSubVolumeMinX->text().toDouble(),
						  nvd._extractSubVolumeMinY->text().toDouble(),
						  nvd._extractSubVolumeMinZ->text().toDouble(),
						  nvd._extractSubVolumeMaxX->text().toDouble(),
						  nvd._extractSubVolumeMaxY->text().toDouble(),
						  nvd._extractSubVolumeMaxZ->text().toDouble());
			VolMagick::Dimension d(nvd._extractSubVolumeBoundingBoxDimX->text().toInt(),
					       nvd._extractSubVolumeBoundingBoxDimY->text().toInt(),
					       nvd._extractSubVolumeBoundingBoxDimZ->text().toInt());
			
			//throw this error before creating the file...
			if(!bb.isWithin(vfi_copy.boundingBox()))
			  throw VolMagick::SubVolumeOutOfBounds("Subvolume bounding box must be within "
								"the bounding box of the original volume.");
			
			VolMagick::createVolumeFile(nvd._filename->text(),
						    bb,
						    d,
						    vfi_copy.voxelTypes(),
						    vfi_copy.numVariables(),
						    vfi_copy.numTimesteps(),
						    vfi_copy.TMin(),
						    vfi_copy.TMax());
			for(unsigned int var = 0; var < vfi_copy.numVariables(); var++)
			  for(unsigned int time = 0; time < vfi_copy.numTimesteps(); time++)
			    {
			      VolMagick::Volume vol;
			      VolMagick::readVolumeFile(vol,vfi_copy.filename(),var,time);
			      vol.sub(bb,d);
			      VolMagick::writeVolumeFile(vol,nvd._filename->text(),var,time);	
			    }
		      }	
		      break;
		    }
		}
	      else
		{
		  VolMagick::createVolumeFile(nvd._filename->text(),
					      vfi_copy.boundingBox(),
					      vfi_copy.dimension(),
					      vfi_copy.voxelTypes(),
					      vfi_copy.numVariables(),
					      vfi_copy.numTimesteps(),
					      vfi_copy.TMin(),
					      vfi_copy.TMax());
		  
		  for(unsigned int var = 0; var < vfi_copy.numVariables(); var++)
		    for(unsigned int time = 0; time < vfi_copy.numTimesteps(); time++)
		      {
			VolMagick::Volume vol;
			VolMagick::readVolumeFile(vol,vfi_copy.filename(),var,time);
			VolMagick::writeVolumeFile(vol,nvd._filename->text(),var,time);	
		      }
		}
		  
	      addVolumeToStack(VolMagick::VolumeFileInfo(nvd._filename->text()));
	    }
	}
      catch(VolMagick::Exception &e)
	{
	  QMessageBox::critical(this,
				"Error",
				QString("%1").arg(e.what()),
				QMessageBox::Ok,QMessageBox::NoButton,QMessageBox::NoButton);
	}
    }
}

void VolumeMaker::openVolumeSlot()
{
  QStringList filenames = QFileDialog::getOpenFileNames("RawIV (*.rawiv);;"
							"RawV (*.rawv);;"
							"MRC (*.mrc);;"
							"INR (*.inr);;"
							"Spider (*.vol *.xmp *.spi);;"
							"All Files (*)",
							QString::null,
							this);
  for(QStringList::iterator it = filenames.begin();
      it != filenames.end();
      ++it)
    {
      if((*it).isEmpty()) continue;
      try
	{
	  addVolumeToStack(VolMagick::VolumeFileInfo(*it));
	}
      catch(VolMagick::Exception &e)
	{
	  QMessageBox::critical(this,
				"Error",
				QString("%1").arg(e.what()),
				QMessageBox::Ok,QMessageBox::NoButton,QMessageBox::NoButton);
	}
    }
}

void VolumeMaker::closeVolumeSlot()
{
  QListViewItemIterator it(_volumeStackList, QListViewItemIterator::Selected);
  QListViewItem *cur;

  if(it.current() == NULL)
    {
      QMessageBox::critical(this,
			    "Error",
			    "Select a volume to close.",
			    QMessageBox::Ok,QMessageBox::NoButton,QMessageBox::NoButton);
      return;
    }

  while((cur=it.current())!=NULL)
    {
      ++it;

      //For some reason, the widget stack will still show the deleted VolumeInterface
      //if we don't raise a widget to replace it...
      if(cur->itemBelow())
	_volumeStack->raiseWidget(_itemToInterface[cur->itemBelow()]);
      else if(cur->itemAbove())
	_volumeStack->raiseWidget(_itemToInterface[cur->itemAbove()]);

      VolumeInterface *vi = _itemToInterface[cur];
      delete vi;
      delete cur;
      _interfaceToItem.erase(vi);
      _itemToInterface.erase(cur);
    }
}

void VolumeMaker::addVolumeToStack(const VolMagick::VolumeFileInfo &vfi)
{
  QListViewItem *volitem = _volumeStackList->findItem(vfi.filename(),1);
  if(volitem)
    {
      if(QMessageBox::information(this,
				  "File already opened",
				  QString("Close and re-open file %1?").arg(QFileInfo(vfi.filename()).fileName()),
				  QMessageBox::Cancel,
				  QMessageBox::Ok) == QMessageBox::Cancel)
	    return;


      delete _itemToInterface[volitem];
      delete volitem;
      _interfaceToItem.erase(_itemToInterface[volitem]);
      _itemToInterface.erase(volitem);
    }

  VolumeInterface *vi = new VolumeInterface(vfi,_mainFrame);
  _volumeStack->addWidget(vi);
  _volumeStack->raiseWidget(vi);
  volitem = new QListViewItem(_volumeStackList,QFileInfo(vfi.filename()).fileName(),vfi.filename());

  _itemToInterface[volitem] = vi;
  _interfaceToItem[vi] = volitem;
}

void VolumeMaker::raiseSelected()
{
  QListViewItemIterator it(_volumeStackList, QListViewItemIterator::Selected);
  _volumeStack->raiseWidget(_itemToInterface[it.current()]);
}
