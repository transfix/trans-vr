/*
  Copyright 2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
                 Alex Rand <arand@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeRover.

  VolumeRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <CVC/App.h>
#include <VolumeRover2/AnisotropicDiffusionDialog.h>

#include <VolMagick/VolMagick.h>

#include <QFileDialog>
#include <QMessageBox>

#include "ui_AnisotropicDiffusionDialog.h"

#include <iostream> 
using namespace std;

// 10/08/2011 -- transfix -- Defaulting to zoomed_volume, and adding extensions from VolMagick
AnisotropicDiffusionDialog::AnisotropicDiffusionDialog(QWidget *parent,Qt::WFlags flags) 
  : QDialog(parent, flags) {

  int idx = -1;

  _ui = new Ui::AnisotropicDiffusionDialog;
  _ui->setupUi(this);
  
  
  // connect slots and signals?
  connect(_ui->OutputFileButton,
	  SIGNAL(clicked()),
	  SLOT(OutputFileSlot()));
  
  _ui->IterationsEdit->insert("20");
  
  std::vector<std::string> keys;
  
  std::vector<std::string> vfiKeys = cvcapp.data<VolMagick::VolumeFileInfo>();
  std::vector<std::string> volKeys = cvcapp.data<VolMagick::Volume>();
  keys.insert(keys.end(),
	      vfiKeys.begin(),vfiKeys.end());
  keys.insert(keys.end(),
	      volKeys.begin(),volKeys.end());
  
  if(keys.empty())
    {
      QMessageBox::information(this, tr("Anisotropic Diffusion"),
                                 tr("No volume loaded."), QMessageBox::Ok);
      return;
    }
  
  BOOST_FOREACH(std::string key, keys)
    _ui->VolumeList->addItem(QString::fromStdString(key));    
  
  std::vector<std::string> extensions = VolMagick::VolumeFile_IO::getExtensions();
  BOOST_FOREACH(std::string ext, extensions)
    _ui->FileTypeComboBox->addItem(QString::fromStdString(ext));
  
  //default to .cvc type if available
  idx = _ui->FileTypeComboBox->findText(".cvc");
  if(idx != -1)
    _ui->FileTypeComboBox->setCurrentItem(idx);
  else
    {
      //if .cvc isn't available (no HDF5), then default to .rawiv
      idx = _ui->FileTypeComboBox->findText(".rawiv");
      if(idx != -1)
        _ui->FileTypeComboBox->setCurrentItem(idx);
    }

  //Default to zoomed_volume if it is in the list
  idx = _ui->VolumeList->findText("zoomed_volume");
  if(idx != -1)
    {
      _ui->tabWidget->setCurrentIndex(1); //preview tab
      _ui->VolumeList->setCurrentItem(idx);
      _ui->DataSetName->setText("zoomed_volume");
    }
}

AnisotropicDiffusionDialog::~AnisotropicDiffusionDialog() {
  delete _ui;
}

void AnisotropicDiffusionDialog::OutputFileSlot() {
  _ui->Output->setText(QFileDialog::getSaveFileName(this,
						    "Results File"));
}

class AnisotropicDiffusionThread
{
public:
  AnisotropicDiffusionThread(int iterations,
                             const std::string& volSelected,
                             const std::string& output,
                             const std::string& dataset,
                             int currentIndex,
                             const std::string& fileType)
    : _iterations(iterations), _volSelected(volSelected),
      _output(output), _dataset(dataset), 
      _currentIndex(currentIndex), _fileType(fileType) {}

  void operator()()
  {
    CVC::ThreadFeedback feedback;

    if (_output.empty()) {
      _output = _volSelected+"_anisotropic_diff";
    }
    if (_dataset.empty()) {
      _dataset = _volSelected+"_anisotropic_diff";
    }
    

    // read in the data if necessary
    if(cvcapp.isData<VolMagick::VolumeFileInfo>(_volSelected))
      {
        VolMagick::VolumeFileInfo vfi = cvcapp.data<VolMagick::VolumeFileInfo>(_volSelected);

        if (_currentIndex == 0) {
          if (_output.substr(_output.find_last_of(".")) != _fileType) {
            _output = _output + _fileType;   
          }

          VolMagick::createVolumeFile(_output,
                                      vfi.boundingBox(),
                                      vfi.dimension(),
                                      vfi.voxelTypes(),
                                      vfi.numVariables(),
                                      vfi.numTimesteps(),
                                      vfi.TMin(),
                                      vfi.TMax());
        }
      
        // run anisotropic diffusion
        for(unsigned int var=0; var<vfi.numVariables(); var++) {
          for(unsigned int time=0; time<vfi.numTimesteps(); time++) {
            VolMagick::Volume vol;
	  
            readVolumeFile(vol,vfi.filename(),var,time);
            vol.anisotropicDiffusion(_iterations);
	  
            if (_currentIndex == 0) {
              writeVolumeFile(vol,_output,var,time);
            } else if (_currentIndex == 1) {
              // put the dataset in the list
              cvcapp.data(_dataset,vol);		  
            }
          }
        }
      }
    else if(cvcapp.isData<VolMagick::Volume>(_volSelected))
      {
        VolMagick::Volume vol = cvcapp.data<VolMagick::Volume>(_volSelected);
        vol.anisotropicDiffusion(_iterations);
      
        if (_currentIndex == 0) {
          //if _output not set, overwrite the input volume data
          if(_output.empty())
            cvcapp.data(_volSelected, vol);
          else {
            _output = _output + _fileType;
            cvcapp.data(_output, vol);
          }
        } else if (_currentIndex == 1) {
	  cvcapp.data(_dataset,vol);
        }      
      }
  }

private:
  int _iterations;
  std::string _volSelected;
  std::string _output;
  std::string _dataset;
  int _currentIndex;
  std::string _fileType;
};

void AnisotropicDiffusionDialog::RunAnisotropicDiffusion() {
  // get parameters
  int iterations = _ui->IterationsEdit->displayText().toInt();
  
  // find the volume files to work with
  std::string volSelected = _ui->VolumeList->currentText().toStdString();
 
  std::string output = _ui->Output->displayText().toStdString();
  std::string dataset = _ui->DataSetName->displayText().toStdString();     

  cvcapp.startThread("anisotropic_diffusion_thread_" + volSelected,
                     AnisotropicDiffusionThread(iterations,
                                                volSelected,
                                                output,
                                                dataset,
                                                _ui->tabWidget->currentIndex(),
                                                _ui->FileTypeComboBox->currentText().toStdString()));
}


