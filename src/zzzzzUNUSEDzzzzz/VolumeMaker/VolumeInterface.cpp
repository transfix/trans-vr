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

/* $Id: VolumeInterface.cpp 1527 2010-03-12 22:10:16Z transfix $ */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iterator>
#include <boost/filesystem.hpp>
#include <qstring.h>
#include <qlabel.h>
#include <qlineedit.h>
#include <qlistview.h>
#include <qfileinfo.h>
#include <VolMagick/VolMagick.h>
#include <VolMagick/endians.h>
#include <qpushbutton.h>
#include <qmessagebox.h>
#include <qbuttongroup.h>
#include <qcombobox.h>
#include <qcheckbox.h>
#include <VolumeMaker/VolumeInterface.h>
#include <VolumeMaker/DimensionModify.h>
#include <VolumeMaker/BoundingBoxModify.h>
#include <VolumeMaker/ImportData.h>
#include <VolumeMaker/RemapVoxels.h>

#include "addvariablebase.h" //just use the base dialogs for now... might need to extend later
#include "addtimestepbase.h"
#include "editvariablebase.h"

VolumeInterface::VolumeInterface( const VolMagick::VolumeFileInfo &vfi,
				  QWidget* parent, const char* name, WFlags f)
  : VolumeInterfaceBase(parent,name,f)
{
  setInterfaceInfo(vfi);

//   _addVariable->setDisabled(true);
//   _deleteVariable->setDisabled(true);
//   _addTimestep->setDisabled(true);
//   _deleteTimestep->setDisabled(true);
}

VolumeInterface::~VolumeInterface() {}

void VolumeInterface::setInterfaceInfo(const VolMagick::VolumeFileInfo &vfi)
{
  _numVars->setText(QString("%1").arg(vfi.numVariables()));
  _numTimesteps->setText(QString("%1").arg(vfi.numTimesteps()));
  _dimensionX->setText(QString("%1").arg(vfi.XDim()));
  _dimensionY->setText(QString("%1").arg(vfi.YDim()));
  _dimensionZ->setText(QString("%1").arg(vfi.ZDim()));
  _boundingBoxMinX->setText(QString("%1").arg(vfi.XMin()));
  _boundingBoxMinY->setText(QString("%1").arg(vfi.YMin()));
  _boundingBoxMinZ->setText(QString("%1").arg(vfi.ZMin()));
  _boundingBoxMaxX->setText(QString("%1").arg(vfi.XMax()));
  _boundingBoxMaxY->setText(QString("%1").arg(vfi.YMax()));
  _boundingBoxMaxZ->setText(QString("%1").arg(vfi.ZMax()));
  _spanX->setText(QString("%1").arg(vfi.XSpan()));
  _spanY->setText(QString("%1").arg(vfi.YSpan()));
  _spanZ->setText(QString("%1").arg(vfi.ZSpan()));

  _variableList->clear();

  //get the number of digits needed for variable and timestep indices
  int vd, var_digits; for(vd = vfi.numVariables(), var_digits=0; vd > 0; vd /= 10, var_digits++);
  int td, time_digits; for(td = vfi.numTimesteps(), time_digits=0; td > 0; td /= 10, time_digits++);

  //build the sprintf format specification
  QString var_fmt(QString("%0") + QString("%1").arg(var_digits) + QString("d"));
  QString time_fmt(QString("%0") + QString("%1").arg(time_digits) + QString("d"));

  double volmin = vfi.min(), volmax = vfi.max();
  for(unsigned int i = 0; i<vfi.numVariables(); i++)
    {
      double varmin = vfi.min(i,0), varmax = vfi.max(i,0);
      QListViewItem *var = new QListViewItem(_variableList,
					     QString("(%1) %2").arg(QString("%1").sprintf(var_fmt.ascii(),i)).arg(vfi.name(i)),
					     QString("%1").arg(vfi.voxelTypeStr(i)));
      for(unsigned int j = 0; j<vfi.numTimesteps(); j++)
	{
	  new QListViewItem(var,
			    "Timestep",
			    QString("%1").arg(vfi.voxelTypeStr(i)),
			    QString("%1").arg(vfi.min(i,j)),
			    QString("%1").arg(vfi.max(i,j)),
			    QString("%1").sprintf(var_fmt.ascii(),i),
			    QString("%1").sprintf(time_fmt.ascii(),j));

	  if(varmin > vfi.min(i,j)) varmin = vfi.min(i,j);
	  if(varmax < vfi.max(i,j)) varmax = vfi.max(i,j);
	}
      var->setText(2,QString("%1").arg(varmin));
      var->setText(3,QString("%1").arg(varmax));
      var->setText(4,QString("%1").arg(i));
      _variableList->setOpen(var,true);

      if(volmin > varmin) volmin = varmin;
      if(volmax < varmax) volmax = varmax;
    }
  
  _minValue->setText(QString("%1").arg(volmin));
  _maxValue->setText(QString("%1").arg(volmax));

  _vfi = vfi;
}

void VolumeInterface::dimensionModifySlot()
{
  DimensionModify dm;

  dm._dimensionX->setText(QString("%1").arg(_vfi.XDim()));
  dm._dimensionY->setText(QString("%1").arg(_vfi.YDim()));
  dm._dimensionZ->setText(QString("%1").arg(_vfi.ZDim()));

  if(dm.exec() == QDialog::Accepted)
    {
      //get a random temporary filename that's not in use
      QString filename_base = QFileInfo(_vfi.filename()).baseName();
      QString filename_ext = QFileInfo(_vfi.filename()).extension();
      
      QString filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;
      while(QFileInfo(filename).exists())
	filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;

      qDebug("temp filename: %s",filename.ascii());
      
      try
	{
	  //create a volume file with enough space for a resize
	  VolMagick::createVolumeFile(filename,
				      _vfi.boundingBox(),
				      VolMagick::Dimension(dm._dimensionX->text().toInt(),
							   dm._dimensionY->text().toInt(),
							   dm._dimensionZ->text().toInt()),
				      _vfi.voxelTypes(),
				      _vfi.numVariables(),
				      _vfi.numTimesteps(),
				      _vfi.TMin(),_vfi.TMax());
      
	  for(unsigned int var=0; var<_vfi.numVariables(); var++)
	    for(unsigned int time=0; time<_vfi.numTimesteps(); time++)
	      {
		VolMagick::Volume vol;
		readVolumeFile(vol,_vfi.filename(),var,time);
		vol.resize(VolMagick::Dimension(dm._dimensionX->text().toInt(),
						dm._dimensionY->text().toInt(),
						dm._dimensionZ->text().toInt()));
		vol.desc(_vfi.name(var));
		writeVolumeFile(vol,filename,var,time);
	      }

	  //now replace the old volume file with the new resized volume
	  boost::filesystem::remove(_vfi.filename());
	  boost::filesystem::copy_file(filename.ascii(),
				       _vfi.filename());
	  boost::filesystem::remove(filename.ascii());
	  _vfi.read(_vfi.filename()); //re-read the volume info
	  setInterfaceInfo(_vfi);
      	}
      catch(VolMagick::Exception &e)
	{
	  QMessageBox::critical(this,
				"Error",
				QString("%1").arg(e.what()),
				QMessageBox::Ok,QMessageBox::NoButton,QMessageBox::NoButton);
	  if(QFileInfo(filename).exists())
	    boost::filesystem::remove(filename.ascii());
	}
    }
}

void VolumeInterface::boundingBoxModifySlot()
{
  BoundingBoxModify bbm;

  bbm._boundingBoxMinX->setText(QString("%1").arg(_vfi.XMin()));
  bbm._boundingBoxMinY->setText(QString("%1").arg(_vfi.YMin()));
  bbm._boundingBoxMinZ->setText(QString("%1").arg(_vfi.ZMin()));
  bbm._boundingBoxMaxX->setText(QString("%1").arg(_vfi.XMax()));
  bbm._boundingBoxMaxY->setText(QString("%1").arg(_vfi.YMax()));
  bbm._boundingBoxMaxZ->setText(QString("%1").arg(_vfi.ZMax()));

  //TODO: this kind of sucks, we should be able to change the 
  //bounding box directly instead of having to copy stuff

  if(bbm.exec() == QDialog::Accepted)
    {
      //get a random temporary filename that's not in use
      QString filename_base = QFileInfo(_vfi.filename()).baseName();
      QString filename_ext = QFileInfo(_vfi.filename()).extension();
      
      QString filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;
      while(QFileInfo(filename).exists())
	filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;

      qDebug("temp filename: %s",filename.ascii());

      //create a bounding box object from user input
      VolMagick::BoundingBox newbox;
      if(!bbm._useCenterPoint->isChecked())
	{
	  newbox = VolMagick::BoundingBox(bbm._boundingBoxMinX->text().toDouble(),
					  bbm._boundingBoxMinY->text().toDouble(),
					  bbm._boundingBoxMinZ->text().toDouble(),
					  bbm._boundingBoxMaxX->text().toDouble(),
					  bbm._boundingBoxMaxY->text().toDouble(),
					  bbm._boundingBoxMaxZ->text().toDouble());
	}
      else
	{
	  newbox = _vfi.boundingBox();
	  double bbox_center_x = newbox.minx + (newbox.maxx - newbox.minx)/2.0;
	  double bbox_center_y = newbox.miny + (newbox.maxy - newbox.miny)/2.0;
	  double bbox_center_z = newbox.minz + (newbox.maxz - newbox.minz)/2.0;
	  double target_x = bbm._centerPointX->text().toDouble();
	  double target_y = bbm._centerPointY->text().toDouble();
	  double target_z = bbm._centerPointZ->text().toDouble();
	  double trans_x = target_x - bbox_center_x;
	  double trans_y = target_y - bbox_center_y;
	  double trans_z = target_z - bbox_center_z;
	  newbox.minx += trans_x; newbox.maxx += trans_x;
	  newbox.miny += trans_y; newbox.maxy += trans_y;
	  newbox.minz += trans_z; newbox.maxz += trans_z;
	}

      try
	{
	  //create a volume file with enough space for a resize
	  VolMagick::createVolumeFile(filename,
				      newbox,
				      _vfi.dimension(),
				      _vfi.voxelTypes(),
				      _vfi.numVariables(),
				      _vfi.numTimesteps(),
				      _vfi.TMin(),_vfi.TMax());

	  for(unsigned int var=0; var<_vfi.numVariables(); var++)
	    for(unsigned int time=0; time<_vfi.numTimesteps(); time++)
	      {
		VolMagick::Volume vol;
		VolMagick::readVolumeFile(vol,_vfi.filename(),var,time);
		vol.desc(_vfi.name(var));
		vol.boundingBox(newbox);
		VolMagick::writeVolumeFile(vol,filename,var,time);
	      }

	  //now replace the old volume file with the new resized volume
	  boost::filesystem::remove(_vfi.filename());
	  boost::filesystem::copy_file(filename.ascii(),
				       _vfi.filename());
	  boost::filesystem::remove(filename.ascii());
	  _vfi.read(_vfi.filename()); //re-read the volume info
	  setInterfaceInfo(_vfi);
	}
      catch(VolMagick::Exception &e)
	{
	  QMessageBox::critical(this,
				"Error",
				QString("%1").arg(e.what()),
				QMessageBox::Ok,QMessageBox::NoButton,QMessageBox::NoButton);
	  if(QFileInfo(filename).exists())
	    boost::filesystem::remove(filename.ascii());
	}
    }
}

void VolumeInterface::addTimestepSlot()
{
  AddTimestepBase at;

  int selected_var, selected_time;
  getSelectedVarTime(selected_var,selected_time);
  if(selected_var == -1 || selected_time == -1)
    {
      QMessageBox::critical( this, "Error", "Select a variable's timestep" );
      return;
    }

  if(at.exec() == QDialog::Accepted)
    {
      //get a random temporary filename that's not in use
      QString filename_base = QFileInfo(_vfi.filename()).baseName();
      QString filename_ext = QFileInfo(_vfi.filename()).extension();
      
      QString filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;
      while(QFileInfo(filename).exists())
	filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;

      qDebug("temp filename: %s",filename.ascii());

      try
	{
	  //create a volume file with an extra timestep
	  VolMagick::createVolumeFile(filename,
				      _vfi.boundingBox(),
				      _vfi.dimension(),
				      _vfi.voxelTypes(),
				      _vfi.numVariables(),
				      _vfi.numTimesteps()+1,
				      _vfi.TMin(),_vfi.TMax()+_vfi.TSpan());

	  for(unsigned int var=0; var<_vfi.numVariables(); var++)
	    {
	      std::vector<int> time_indices;
	      for(unsigned int time=0; time<_vfi.numTimesteps(); time++)
		time_indices.push_back(static_cast<int>(time));
	      time_indices.insert(time_indices.begin()+
				  selected_time+
				  (at._beforeOrAfterGroup->selectedId() == 0 ? 0 : 1),
				  -1);
	      for(std::vector<int>::iterator time = time_indices.begin();
		  time != time_indices.end();
		  time++)
		{
		  if(*time != -1)
		    {
		      VolMagick::Volume vol;
		      VolMagick::readVolumeFile(vol,
						_vfi.filename(),
						var,
						static_cast<unsigned int>(*time));
		      vol.desc(_vfi.name(var));
		      VolMagick::writeVolumeFile(vol,
						 filename,
						 var,
						 std::distance(time_indices.begin(),
							       time));
		    }
		}
	    }

	  //now replace the old volume file with the new resized volume
	  boost::filesystem::remove(_vfi.filename());
	  boost::filesystem::copy_file(filename.ascii(),
				       _vfi.filename());
	  boost::filesystem::remove(filename.ascii());
	  _vfi.read(_vfi.filename()); //re-read the volume info
	  setInterfaceInfo(_vfi);
	}
      catch(VolMagick::Exception &e)
	{
	  QMessageBox::critical(this,
				"Error",
				QString("%1").arg(e.what()),
				QMessageBox::Ok,
				QMessageBox::NoButton,
				QMessageBox::NoButton);
	  if(QFileInfo(filename).exists())
	    boost::filesystem::remove(filename.ascii());
	}
    }
}

void VolumeInterface::addVariableSlot()
{
  AddVariableBase av;

  int selected_var, selected_time;
  getSelectedVarTime(selected_var,selected_time);
  if(selected_var == -1)
    {
      QMessageBox::critical( this, "Error", "Select a variable" );
      return;
    }

  if(av.exec() == QDialog::Accepted)
    {
      //get a random temporary filename that's not in use
      QString filename_base = QFileInfo(_vfi.filename()).baseName();
      QString filename_ext = QFileInfo(_vfi.filename()).extension();
      
      QString filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;
      while(QFileInfo(filename).exists())
	filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;

      qDebug("temp filename: %s",filename.ascii());

      try
	{
	  //first set up a new voxel type vector to include the new variable's type
	  std::vector<VolMagick::VoxelType> newtypes(_vfi.voxelTypes());
	  newtypes.insert(newtypes.begin()+
			  selected_var+
			  (av._beforeOrAfterGroup->selectedId() == 0 ? 0 : 1),
			  VolMagick::VoxelType(av._dataType->currentItem()));

	  //create a volume file with an extra variable
	  VolMagick::createVolumeFile(filename,
				      _vfi.boundingBox(),
				      _vfi.dimension(),
				      newtypes,
				      _vfi.numVariables()+1,
				      _vfi.numTimesteps(),
				      _vfi.TMin(),_vfi.TMax());

	  std::vector<int> var_indices;
	  for(unsigned int var=0; var<_vfi.numVariables(); var++)
	    var_indices.push_back(static_cast<int>(var));
	  var_indices.insert(var_indices.begin()+
			     selected_var+
			     (av._beforeOrAfterGroup->selectedId() == 0 ? 0 : 1),
			     -1);
	  int newvar = -1;
	  for(std::vector<int>::iterator var = var_indices.begin();
	      var != var_indices.end();
	      var++)
	    {
	      if(*var != -1)
		{
		  VolMagick::Volume vol;
		  for(unsigned int time=0; time<_vfi.numTimesteps(); time++)
		    {
		      VolMagick::readVolumeFile(vol,
						_vfi.filename(),
						static_cast<unsigned int>(*var),
						time);
		      VolMagick::writeVolumeFile(vol,
						 filename,
						 std::distance(var_indices.begin(),var),
						 time);
		    }
		}
	      else
		newvar = static_cast<int>(std::distance(var_indices.begin(),var));
	    }
	  
	  //set the new variable's name in the volume file...
	  //this sucks because we have to write a whole volume object just to set the name...
	  //TODO: consider making changing the variable name in volmagick easier
	  VolMagick::Volume vol(_vfi.dimension(),
				VolMagick::VoxelType(av._dataType->currentItem()),
				_vfi.boundingBox());
	  vol.desc(av._name->text());
	  VolMagick::writeVolumeFile(vol,filename,newvar);

	  //now replace the old volume file with the new resized volume
	  boost::filesystem::remove(_vfi.filename());
	  boost::filesystem::copy_file(filename.ascii(),
				       _vfi.filename());
	  boost::filesystem::remove(filename.ascii());
	  _vfi.read(_vfi.filename()); //re-read the volume info
	  setInterfaceInfo(_vfi);
	}
      catch(VolMagick::Exception &e)
	{
	  QMessageBox::critical(this,
				"Error",
				QString("%1").arg(e.what()),
				QMessageBox::Ok,QMessageBox::NoButton,QMessageBox::NoButton);
	  if(QFileInfo(filename).exists())
	    boost::filesystem::remove(filename.ascii());
	}
    }
}

void VolumeInterface::deleteTimestepSlot()
{
  int selected_var, selected_time;
  getSelectedVarTime(selected_var,selected_time);
  if(selected_var == -1)
    {
      QMessageBox::critical( this, "Error", "Select a variable's timestep" );
      return;
    }

  //get a random temporary filename that's not in use
  QString filename_base = QFileInfo(_vfi.filename()).baseName();
  QString filename_ext = QFileInfo(_vfi.filename()).extension();
  
  QString filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;
  while(QFileInfo(filename).exists())
    filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;
  
  qDebug("temp filename: %s",filename.ascii());

  if(_vfi.numTimesteps() < 2)
    {
      QMessageBox::critical(this,
			    "Error",
			    "Cannot remove any more timesteps!",
			    QMessageBox::Ok,
			    QMessageBox::NoButton,
			    QMessageBox::NoButton);
      return;
    }

  try
    {
      //create a volume file with 1 less timestep
      VolMagick::createVolumeFile(filename,
				  _vfi.boundingBox(),
				  _vfi.dimension(),
				  _vfi.voxelTypes(),
				  _vfi.numVariables(),
				  _vfi.numTimesteps()-1,
				  _vfi.TMin(),_vfi.TMax()-_vfi.TSpan());
      
      for(unsigned int var=0; var<_vfi.numVariables(); var++)
	{
	  std::vector<int> time_indices;
	  for(unsigned int time=0; time<_vfi.numTimesteps(); time++)
	    time_indices.push_back(static_cast<int>(time));
	  time_indices.erase(time_indices.begin()+selected_time);
	  for(std::vector<int>::iterator time = time_indices.begin();
	      time != time_indices.end();
	      time++)
	    {
	      VolMagick::Volume vol;
	      VolMagick::readVolumeFile(vol,
					_vfi.filename(),
					var,
					static_cast<unsigned int>(*time));
	      VolMagick::writeVolumeFile(vol,
					 filename,
					 var,
					 std::distance(time_indices.begin(),
						       time));
	    }
	}

      //now replace the old volume file with the new volume
      boost::filesystem::remove(_vfi.filename());
      boost::filesystem::copy_file(filename.ascii(),
				   _vfi.filename());
      boost::filesystem::remove(filename.ascii());
      _vfi.read(_vfi.filename()); //re-read the volume info
      setInterfaceInfo(_vfi);
    }
  catch(VolMagick::Exception &e)
    {
      QMessageBox::critical(this,
			    "Error",
			    QString("%1").arg(e.what()),
			    QMessageBox::Ok,
			    QMessageBox::NoButton,
			    QMessageBox::NoButton);
      if(QFileInfo(filename).exists())
	boost::filesystem::remove(filename.ascii());
    }
}

void VolumeInterface::deleteVariableSlot()
{
  int selected_var, selected_time;
  getSelectedVarTime(selected_var,selected_time);
  if(selected_var == -1)
    {
      QMessageBox::critical( this, "Error", "Select a variable" );
      return;
    }

  //get a random temporary filename that's not in use
  QString filename_base = QFileInfo(_vfi.filename()).baseName();
  QString filename_ext = QFileInfo(_vfi.filename()).extension();
  
  QString filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;
  while(QFileInfo(filename).exists())
    filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;
  
  qDebug("temp filename: %s",filename.ascii());

  if(_vfi.numVariables() < 2)
    {
      QMessageBox::critical(this,
			    "Error",
			    "Cannot remove any more variables!",
			    QMessageBox::Ok,
			    QMessageBox::NoButton,
			    QMessageBox::NoButton);
      return;
    }

  try
    {
      //first set up a new voxel type vector to erase the selected variable's type
      std::vector<VolMagick::VoxelType> newtypes(_vfi.voxelTypes());
      newtypes.erase(newtypes.begin()+selected_var);

      //create a volume file with 1 less variable
      VolMagick::createVolumeFile(filename,
				  _vfi.boundingBox(),
				  _vfi.dimension(),
				  newtypes,
				  _vfi.numVariables()-1,
				  _vfi.numTimesteps(),
				  _vfi.TMin(),_vfi.TMax());

      std::vector<int> var_indices;
      for(unsigned int var=0; var<_vfi.numVariables(); var++)
	var_indices.push_back(static_cast<int>(var));
      var_indices.erase(var_indices.begin()+selected_var);
      for(std::vector<int>::iterator var = var_indices.begin();
	  var != var_indices.end();
	  var++)
	{
	  VolMagick::Volume vol;
	  for(unsigned int time=0; time<_vfi.numTimesteps(); time++)
	    {
	      VolMagick::readVolumeFile(vol,
					_vfi.filename(),
					static_cast<unsigned int>(*var),
					time);
	      VolMagick::writeVolumeFile(vol,
					 filename,
					 std::distance(var_indices.begin(),var),
					 time);
	    }
	}

      //now replace the old volume file with the new volume
      boost::filesystem::remove(_vfi.filename());
      boost::filesystem::copy_file(filename.ascii(),
				   _vfi.filename());
      boost::filesystem::remove(filename.ascii());
      _vfi.read(_vfi.filename()); //re-read the volume info
      setInterfaceInfo(_vfi);
    }
  catch(VolMagick::Exception &e)
    {
      QMessageBox::critical(this,
			    "Error",
			    QString("%1").arg(e.what()),
			    QMessageBox::Ok,
			    QMessageBox::NoButton,
			    QMessageBox::NoButton);
      if(QFileInfo(filename).exists())
	boost::filesystem::remove(filename.ascii());
    }
}

void VolumeInterface::editVariableSlot()
{
  EditVariableBase ev;

  int selected_var, selected_time;
  getSelectedVarTime(selected_var,selected_time);
  if(selected_var == -1)
    {
      QMessageBox::critical( this, "Error", "Select a variable" );
      return;
    }

  ev._name->setText(_vfi.name(selected_var));
  ev._dataType->setCurrentItem(static_cast<int>(_vfi.voxelTypes(selected_var)));

  if(ev.exec() == QDialog::Accepted)
    {
      //get a random temporary filename that's not in use
      QString filename_base = QFileInfo(_vfi.filename()).baseName();
      QString filename_ext = QFileInfo(_vfi.filename()).extension();
      
      QString filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;
      while(QFileInfo(filename).exists())
	filename = filename_base + "." + QString("%1").arg(rand()) + "." + filename_ext;
      
      qDebug("temp filename: %s",filename.ascii());

      try
	{
	  //first set up a new voxel type vector to include the new variable's type
	  std::vector<VolMagick::VoxelType> newtypes(_vfi.voxelTypes());
	  newtypes[selected_var] = VolMagick::VoxelType(ev._dataType->currentItem());
	  
	  //create a volume file with an edited variable
	  VolMagick::createVolumeFile(filename,
				      _vfi.boundingBox(),
				      _vfi.dimension(),
				      newtypes,
				      _vfi.numVariables(),
				      _vfi.numTimesteps(),
				      _vfi.TMin(),_vfi.TMax());

	  for(unsigned int time = 0; time < _vfi.numTimesteps(); time++)
	    {
	      VolMagick::Volume vol;
	      VolMagick::readVolumeFile(vol,
					_vfi.filename(),
					selected_var,
					time);
	      vol.desc(ev._name->text());
	      vol.voxelType(newtypes[selected_var]);
	      VolMagick::writeVolumeFile(vol,
					 filename,
					 selected_var,
					 time);
	    }

	  //since we created a new volume, we have to re-add all the variable names...
	  //TODO: please make variable name changing easier&quicker!!!!! Need to modify VolMagick library
	  for(unsigned int var = 0; var < _vfi.numVariables(); var++)
	    {
	      if(var != (unsigned int)selected_var)
		{
		  VolMagick::Volume vol;
		  VolMagick::readVolumeFile(vol,_vfi.filename(),var,0);
		  VolMagick::writeVolumeFile(vol,filename,var,0);
		}
	    }

	  //now replace the old volume file with the new resized volume
	  boost::filesystem::remove(_vfi.filename());
	  boost::filesystem::copy_file(filename.ascii(),
				       _vfi.filename());
	  boost::filesystem::remove(filename.ascii());
	  _vfi.read(_vfi.filename()); //re-read the volume info
	  setInterfaceInfo(_vfi);
	}
      catch(VolMagick::Exception &e)
	{
	  QMessageBox::critical(this,
				"Error",
				QString("%1").arg(e.what()),
				QMessageBox::Ok,QMessageBox::NoButton,QMessageBox::NoButton);
	  if(QFileInfo(filename).exists())
	    boost::filesystem::remove(filename.ascii());
	}
    }
}

void VolumeInterface::importDataSlot()
{
  ImportData id;

  int selected_var, selected_time;
  getSelectedVarTime(selected_var,selected_time);
  if(selected_var == -1 || selected_time == -1)
    {
      QMessageBox::critical( this, "Error", "Select a variable's timestep" );
      return;
    }

  if(id.exec() == QDialog::Accepted)
    {
      try
	{
	  if(id._fileTypeGroup->selectedId() == 0) //raw data
	    {
	      VolMagick::Volume rawvol(VolMagick::Dimension(id._dimensionX->text().toInt(),
							    id._dimensionY->text().toInt(),
							    id._dimensionZ->text().toInt()),
				       VolMagick::VoxelType(id._dataType->currentItem()));

	  

	      FILE *fp = fopen(id._importFile->text().ascii(),"rb");
	      if(fp == NULL)
		{
		  QMessageBox::critical( this, "Error", "Could not open import file!" );
		  return;
		}

	      //offset from beginning
	      fseek(fp,id._offset->text().toInt(),SEEK_SET);

	      if(rawvol.XDim()*rawvol.YDim()*rawvol.ZDim() != fread(*rawvol,
								    rawvol.voxelSize(),
								    rawvol.XDim()*rawvol.YDim()*rawvol.ZDim(),
								    fp))
		{
		  QMessageBox::critical( this, "Error", "Read Error!" );
		  return;
		}
	  
	      if(big_endian() && id._endianGroup->selectedId() == 0)
		{
		  size_t i;
		  size_t len = rawvol.XDim()*rawvol.YDim()*rawvol.ZDim();
		  switch(rawvol.voxelType())
		    {
		    case VolMagick::UShort: for(i=0;i<len;i++) SWAP_16(*rawvol+i*rawvol.voxelSize()); break;
		    case VolMagick::UInt:
		    case VolMagick::Float:  for(i=0;i<len;i++) SWAP_32(*rawvol+i*rawvol.voxelSize()); break;
		    case VolMagick::UInt64:
		    case VolMagick::Double: for(i=0;i<len;i++) SWAP_64(*rawvol+i*rawvol.voxelSize()); break;
		    default: break;
		    }
		}
	      else if(!big_endian() && id._endianGroup->selectedId() == 1)
		{
		  size_t i;
		  size_t len = rawvol.XDim()*rawvol.YDim()*rawvol.ZDim();
		  switch(rawvol.voxelType())
		    {
		    case VolMagick::UShort: for(i=0;i<len;i++) SWAP_16(*rawvol+i*rawvol.voxelSize()); break;
		    case VolMagick::UInt:
		    case VolMagick::Float:  for(i=0;i<len;i++) SWAP_32(*rawvol+i*rawvol.voxelSize()); break;
		    case VolMagick::UInt64:
		    case VolMagick::Double: for(i=0;i<len;i++) SWAP_64(*rawvol+i*rawvol.voxelSize()); break;
		    default: break;
		    }
		}

	      if(_vfi.dimension() != rawvol.dimension())
		{
		  if(QMessageBox::information(this,
					      "Size mismatch",
					      "Import data does not match the dimension of volume.  Resize?",
					      QMessageBox::No,
					      QMessageBox::Yes) == QMessageBox::No)
		    return;

		  rawvol.resize(_vfi.dimension());
		}

	      if(_vfi.voxelTypes(selected_var) != rawvol.voxelType())
		{
		  if(QMessageBox::information(this,
					      "Type mismatch",
					      "Import data does not match the type of volume.  Convert?",
					      QMessageBox::No,
					      QMessageBox::Yes) == QMessageBox::No)
		    return;

		  //TODO: allow user to re-map voxel values so they can choose to lose precision or not...
		  rawvol.voxelType(_vfi.voxelTypes(selected_var));
		}

	      rawvol.desc(_vfi.name(selected_var));
	      VolMagick::writeVolumeFile(rawvol,_vfi.filename(),selected_var,selected_time);
	    }
	  else
	    {
	      VolMagick::Volume vol;

	      VolMagick::readVolumeFile(vol,
					id._importFile->text(),
					id._variable->text().toInt(),
					id._timestep->text().toInt());

	      if(_vfi.dimension() != vol.dimension())
		{
		  if(QMessageBox::information(this,
					      "Size mismatch",
					      "Import data does not match the dimension of volume.  Resize?",
					      QMessageBox::No,
					      QMessageBox::Yes) == QMessageBox::No)
		    return;

		  vol.resize(_vfi.dimension());
		}

	      if(_vfi.voxelTypes(selected_var) != vol.voxelType())
		{
		  if(QMessageBox::information(this,
					      "Type mismatch",
					      "Import data does not match the type of volume.  Convert?",
					      QMessageBox::No,
					      QMessageBox::Yes) == QMessageBox::No)
		    return;

		  //TODO: allow user to re-map voxel values so they can choose to lose precision or not...
		  vol.voxelType(_vfi.voxelTypes(selected_var));
		}

	      vol.desc(_vfi.name(selected_var));
	      VolMagick::writeVolumeFile(vol,_vfi.filename(),selected_var,selected_time);
	    }

	  _vfi.read(_vfi.filename()); //re-read the volume info
	  setInterfaceInfo(_vfi);
	}
      catch(VolMagick::Exception &e)
	{
	  QMessageBox::critical(this,
				"Error",
				QString("%1").arg(e.what()),
				QMessageBox::Ok,
				QMessageBox::NoButton,
				QMessageBox::NoButton);
	}
    }
}

void VolumeInterface::remapSlot()
{
  RemapVoxels rv;

  int selected_var, selected_time;
  getSelectedVarTime(selected_var,selected_time);
  if(selected_var == -1 || selected_time == -1)
    {
      QMessageBox::critical( this, "Error", "Select a variable's timestep" );
      return;
    }

  rv._minValue->setText(QString("%1").arg(_vfi.min(selected_var,selected_time)));
  rv._maxValue->setText(QString("%1").arg(_vfi.max(selected_var,selected_time)));

  if(rv.exec() == QDialog::Accepted)
    {
      try
	{
	  VolMagick::Volume vol;
	  VolMagick::readVolumeFile(vol,_vfi.filename(),selected_var,selected_time);
	  vol.map(rv._minValue->text().toDouble(),
		  rv._maxValue->text().toDouble());
	  VolMagick::writeVolumeFile(vol,_vfi.filename(),selected_var,selected_time);
	  _vfi.read(_vfi.filename()); //re-read the volume info
	  setInterfaceInfo(_vfi);
	}
      catch(VolMagick::Exception &e)
	{
	  QMessageBox::critical(this,
				"Error",
				QString("%1").arg(e.what()),
				QMessageBox::Ok,
				QMessageBox::NoButton,
				QMessageBox::NoButton);
	}
    }
}

void VolumeInterface::getSelectedVarTime(int &var, int &time)
{
  if(_variableList->selectedItem() == NULL)
    {
      var = time = -1;
      return;
    }

//   if(_variableList->selectedItem()->text(0) != "Timestep")
//     {
//       var = time = -1;
//       return;
//     }

  var = _variableList->selectedItem()->text(4).toInt();
  time = _variableList->selectedItem()->text(5).isEmpty() ? -1 :
    _variableList->selectedItem()->text(5).toInt();
}
