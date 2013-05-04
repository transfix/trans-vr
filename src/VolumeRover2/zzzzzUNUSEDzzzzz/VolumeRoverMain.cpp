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

/* $Id: VolumeRoverMain.cpp 4087 2011-05-06 14:54:32Z arand $ */

#include <qglobal.h>

#if QT_VERSION < 0x040000
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
#include <qmenubar.h>
#include <qpopupmenu.h>
#include <qaction.h>
#include "volumerovermainbase.Qt3.h" //generated from the .Qt3.ui file
#else
#include <QLineEdit>
#include <QStackedWidget>
#include <QTabWidget>
#include <QGridLayout>
#include <QFileInfo>
#include <QFileDialog>
#include <QButtonGroup>
#include <QMessageBox>
#include <QComboBox>
#include <QCheckBox>
#include <QTreeWidget>
#include <QTreeWidgetItemIterator>
#include <QStringList>
#include <QMenu>
#include <QMenuBar>
#include <QList>
#include "ui_VolumeRoverMain.h" //generated from the .Qt4.ui file
#endif

#include <cvcraw_geometry/cvcraw_geometry.h>
#include <VolumeRover2/VolumeRoverMain.h>
#include <VolumeRover2/NewVolumeDialog.h>
#include <VolumeRover2/VolumeInterface.h>
#include <VolumeRover2/VolumeViewerPage.h>

#include "VolumeRover/bilateralfilterdialog.Qt3.h"
#include <Filters/OOCBilateralFilter.h>

#include <cvcalgo/cvcalgo.h>

#ifdef VOLUMEGRIDROVER
#include <VolumeGridRover/VolumeGridRover.h>
#endif

VolumeRoverMain::VolumeRoverMainPtr VolumeRoverMain::_instance; 

VolumeRoverMain::VolumeRoverMainPtr VolumeRoverMain::instancePtr()
{
  if(!_instance)
    {
      _instance.reset(new VolumeRoverMain);

      //core volrover libs.. in the future, call this on plugin load
      //instead of when main window is instantiated.
      cvcalgo::init();
    }
  return _instance;
}

VolumeRoverMain& VolumeRoverMain::instance()
{
  return *instancePtr();
}

VolumeRoverMain::VolumeRoverMain(QWidget* parent, 
#if QT_VERSION < 0x040000
                                 const char* name, WFlags f
#else
                                 Qt::WFlags flags
#endif
                                 )
  : QMainWindow(parent,
#if QT_VERSION < 0x040000
                name,f
#else
                flags
#endif
                ),
    _mainTabs(NULL),
    _volumeStack(NULL),
    _volumeStackList(NULL),
    _volumeViewerPage(NULL),
#ifdef VOLUMEGRIDROVER
    _volumeGridRover(NULL),
#endif
    _ui(NULL)
{
#if QT_VERSION < 0x040000
  _ui = new VolumeRoverMainBase(this);
  setCentralWidget(_ui);
#else
  _centralWidget = new QWidget;
  setCentralWidget(_centralWidget);
  _ui = new Ui::VolumeRoverMain;
  _ui->setupUi(_centralWidget);
#endif

  QGridLayout *mainLayout(new QGridLayout(_ui->_mainFrame));
  _mainTabs = new QTabWidget(_ui->_mainFrame);
  mainLayout->addWidget(_mainTabs,0,0);

  QWidget *tabpage = new QWidget(_mainTabs);
  QGridLayout *tabPageLayout = new QGridLayout(tabpage);
#if QT_VERSION < 0x040000
  tabPageLayout->setColStretch(0,3);
  tabPageLayout->setColStretch(1,2);
#else
  tabPageLayout->setColumnStretch(0,3);
  tabPageLayout->setColumnStretch(1,2);
#endif

#if QT_VERSION < 0x040000
  _volumeStackList = new QListView(tabpage);
  _volumeStackList->addColumn("File Name");
  _volumeStackList->addColumn("Full Path");
  _volumeStackList->setSorting(-1);
  _volumeStackList->setSelectionMode(QListView::Extended);
  connect(_volumeStackList,
	  SIGNAL(selectionChanged()),
	  SLOT(raiseSelected()));
#else
  _volumeStackList = new QTreeWidget(tabpage);
  QStringList labels;
  labels << tr("File Name") << tr("Full Path");
  _volumeStackList->setHeaderLabels(labels);
  _volumeStackList->setSortingEnabled(false);
  _volumeStackList->setSelectionMode(QAbstractItemView::ExtendedSelection);
  connect(_volumeStackList,
	  SIGNAL(itemSelectionChanged()),
	  SLOT(raiseSelected()));
#endif
  tabPageLayout->addWidget(_volumeStackList,0,0);
  _volumeStackList->show();
  
#if QT_VERSION < 0x040000
  _volumeStack = new QWidgetStack(tabpage);
  tabPageLayout->addWidget(_volumeStack,0,1);
  _volumeStack->show();
  connect(_volumeStack,
	  SIGNAL(aboutToShow(int)),
	  SLOT(loadShownVolume(int)));
#else
  _volumeStack = new QStackedWidget(tabpage);
  tabPageLayout->addWidget(_volumeStack,0,1);
  _volumeStack->show();
  connect(_volumeStack,
          SIGNAL(currentChanged(int)),
          SLOT(loadShownVolume(int)));
#endif

  _mainTabs->addTab(tabpage,"Attributes");

  _volumeViewerPage = new VolumeViewerPage(_mainTabs);
  _mainTabs->addTab(_volumeViewerPage,"Viewer");
#if QT_VERSION < 0x040000
  _mainTabs->setCurrentPage(_mainTabs->indexOf(_volumeViewerPage));
#else
  _mainTabs->setCurrentIndex(_mainTabs->indexOf(_volumeViewerPage));
#endif

  //Populate the main window menu
#if QT_VERSION < 0x040000
  QMenuBar   *mainMenu = menuBar();
  QPopupMenu *fileMenu = new QPopupMenu;
  fileMenu->insertItem("&New",  this, SLOT(newVolumeSlot()), CTRL+Key_N);
  fileMenu->insertItem("&Open", this, SLOT(openVolumeSlot()),  CTRL+Key_O);
  fileMenu->insertItem("&Close", this, SLOT(closeVolumeSlot()),  CTRL+Key_C);
  fileMenu->insertSeparator();
  fileMenu->insertItem("&Quit", this, SLOT(close()));
  mainMenu->insertItem("&File", fileMenu );

  QPopupMenu *viewMenu = new QPopupMenu;
  mainMenu->insertItem("&View",viewMenu);

#ifdef VOLUMEGRIDROVER
  //Detached volume grid rover... should we bother having an option for
  //adding it to the tabs? -- Joe R. -- 20100812
  _volumeGridRover = new VolumeGridRover(NULL);
  QAction *showVolumeGridRover = new QAction(this,"showVolumeGridRover");
  showVolumeGridRover->setToggleAction(true);
  showVolumeGridRover->addTo(viewMenu);
  connect(showVolumeGridRover,
          SIGNAL(toggled(bool)),
          this,
          SLOT(toggleVolumeGridRoverSlot(bool)));
  showVolumeGridRover->setText( tr("&Show VolumeGridRover") );
  showVolumeGridRover->setMenuText( tr("&Show VolumeGridRover") );
#endif

  QPopupMenu *utilsMenu = new QPopupMenu;
  mainMenu->insertItem("&Utilities",utilsMenu);
#else
  QMenu *fileMenu = menuBar()->addMenu(tr("&File"));
  fileMenu->addAction(tr("&New"), this, SLOT(newVolumeSlot()));
  fileMenu->addAction(tr("&Open"), this, SLOT(openVolumeSlot()));
  fileMenu->addAction(tr("&Close"), this, SLOT(closeVolumeSlot()));
  fileMenu->addSeparator();
  fileMenu->addAction(tr("&Quit"), this, SLOT(close()));

  QMenu *toolsMenu = menuBar()->addMenu(tr("&Tools"));
  toolsMenu->addAction(tr("&Bilateral Filter"), this, SLOT(bilateralFilterSlot()));

#endif  


  _selectedObject = _objects.end();
}

VolumeRoverMain::~VolumeRoverMain() 
{
  delete _ui;
#ifdef VOLUMEGRIDROVER
  delete _volumeGridRover;
#endif
}

#ifdef VOLUMEGRIDROVER
void VolumeRoverMain::toggleVolumeGridRoverSlot(bool show)
{
  if(_volumeGridRover)
    {
      if(show) _volumeGridRover->show();
      else _volumeGridRover->hide();
    }
}
#endif

void VolumeRoverMain::newVolumeSlot()
{
  //qDebug("newvolumeslot");

  NewVolumeDialog nvd;

  if(nvd.exec() == QDialog::Accepted)
    {
      try
	{
	  if(nvd.createNewVolume()) //new volume
	    {
              //We really need a way to create volumes without having to instantiate
              //a volume in memory!  The main reason I'm doing this below is because
              //the current createVolumeFile doesn't take variable names... - Joe R. 6/25/2010

	      //output the volume
	      VolMagick::Volume vol(nvd.dimension(),
				    nvd.variableType(),
				    nvd.boundingBox());
	      vol.desc(nvd.variableName());
	      VolMagick::createVolumeFile(nvd.filename(),
					  vol.boundingBox(),
					  vol.dimension(),
					  std::vector<VolMagick::VoxelType>(1,vol.voxelType()));
	      VolMagick::writeVolumeFile(vol,nvd.filename());
	  
	      //now read it's info
	      VolMagick::VolumeFileInfo vfi(nvd.filename());
	  
	      addVolumeToStack(vfi);
	    }
	  else //copy volume
	    {
	      VolMagick::VolumeFileInfo vfi_copy(nvd.volumeCopyFilename());

	      VolMagick::BoundingBox bb = vfi_copy.boundingBox();
	      
	      if(nvd.extractSubVolume()) //extract sub volume
		{
		  switch(nvd.extractSubVolumeMethod())
		    {
		    case NewVolumeDialog::INDICES: //using image indices
		      {
                        VolMagick::IndexBoundingBox ibb =
                          nvd.extractIndexSubVolume();

			VolMagick::uint64 min_index[3] =
			  {
                            ibb.XMin(), ibb.YMin(), ibb.ZMin()
			  };
			
			VolMagick::uint64 max_index[3] =
			  {
                            ibb.XMax(), ibb.YMax(), ibb.ZMax()
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
			VolMagick::createVolumeFile(nvd.filename(),
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
			      VolMagick::writeVolumeFile(vol,nvd.filename(),var,time);
			    }
		      }
		      break;
		    case NewVolumeDialog::BOUNDING_BOX: //using bounding box
		      {
                        VolMagick::BoundingBox bb = nvd.extractSubVolumeBoundingBox();
                        VolMagick::Dimension d = nvd.extractSubVolumeDimension();
			
			//throw this error before creating the file...
			if(!bb.isWithin(vfi_copy.boundingBox()))
			  throw VolMagick::SubVolumeOutOfBounds("Subvolume bounding box must be within "
								"the bounding box of the original volume.");
			
			VolMagick::createVolumeFile(nvd.filename(),
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
			      VolMagick::writeVolumeFile(vol,nvd.filename(),var,time);
			    }
		      }	
		      break;
		    }
		}
	      else
		{
		  VolMagick::createVolumeFile(nvd.filename(),
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
			VolMagick::writeVolumeFile(vol,nvd.filename(),var,time);
		      }
		}
		  
	      addVolumeToStack(VolMagick::VolumeFileInfo(nvd.filename()));
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

void VolumeRoverMain::openVolumeSlot()
{
#if QT_VERSION < 0x040000
  QStringList filenames = QFileDialog::getOpenFileNames("RawIV (*.rawiv);;"
							"RawV (*.rawv);;"
							"MRC (*.mrc);;"
							"cvc-raw geometry (*.raw *.rawn *.rawc *.rawnc);;"
							"All Files (*)",
							QString::null,
							this);
#else
  QStringList filenames = QFileDialog::getOpenFileNames(this,
                                                        "Open File",
                                                        QString::null,
                                                        "RawIV (*.rawiv);;"
							"RawV (*.rawv);;"
							"MRC (*.mrc);;"
							"cvc-raw geometry (*.raw *.rawn *.rawc *.rawnc);;"
							"All Files (*)");
#endif

  for(QStringList::iterator it = filenames.begin();
      it != filenames.end();
      ++it)
    {
      if((*it).isEmpty()) continue;

#if QT_VERSION < 0x040000
      std::string cur((*it).ascii());
#else
      std::string cur((*it).toAscii());
#endif

      //TODO: fix file reading below, this is stupid
      try
	{
	  try
	    {
	      addVolumeToStack(VolMagick::VolumeFileInfo(cur));
	    }
	  catch(VolMagick::Exception& e) //error reading, lets try reading as geometry
	    {
	      cvcraw_geometry::geometry_t geom;
	      cvcraw_geometry::read(geom,cur);
	      typedef VolumeViewer::scene_geometry_t scene_geometry_t;
	      scene_geometry_t::render_mode_t mode = scene_geometry_t::TRIANGLES;

	      if(!geom.lines.empty())
		mode = scene_geometry_t::LINES;
	      if(!geom.tris.empty())
		mode = scene_geometry_t::TRIANGLES;
	      if(!geom.quads.empty())
		mode = scene_geometry_t::QUADS;

	      if(!geom.boundary.empty() && !geom.tris.empty())
		{
		  geom = geom.generate_wire_interior();
		  mode = scene_geometry_t::TETRA;
		}
	      else if(!geom.boundary.empty() && !geom.quads.empty())
		{
		  geom = geom.generate_wire_interior();
		  mode = scene_geometry_t::HEXA;
		}

	      _volumeViewerPage->thumbnailViewer()->addGeometry(geom,cur,mode);
	      _volumeViewerPage->subvolumeViewer()->geometries() =
		_volumeViewerPage->thumbnailViewer()->geometries();
	    }
	}
      catch(std::exception& e)
	{
	  QMessageBox::critical(this,
				"Error",
				QString("Error loading file '%1'").arg(*it),
				QMessageBox::Ok,QMessageBox::NoButton,QMessageBox::NoButton);
	}
    }
}

void VolumeRoverMain::closeVolumeSlot()
{
#if QT_VERSION < 0x040000
  QListViewItemIterator it(_volumeStackList, QListViewItemIterator::Selected);
  QListViewItem *cur;
#else
  QTreeWidgetItemIterator it(_volumeStackList, QTreeWidgetItemIterator::Selected);
  QTreeWidgetItem *cur;
#endif

#if QT_VERSION < 0x040000
  if(it.current() == NULL)
#else
  if(*it == NULL)
#endif
    {
      QMessageBox::critical(this,
			    "Error",
			    "Select a volume to close.",
			    QMessageBox::Ok,QMessageBox::NoButton,QMessageBox::NoButton);
      return;
    }

#if QT_VERSION < 0x040000
  while((cur=it.current())!=NULL)
#else
  while((cur=*it)!=NULL)
#endif
    {
      ++it;

      //For some reason, the widget stack will still show the deleted VolumeInterface
      //if we don't raise a widget to replace it...
#if QT_VERSION < 0x040000
      if(cur->itemBelow())
	_volumeStack->raiseWidget(_itemToInterface[cur->itemBelow()]);
      else if(cur->itemAbove())
	_volumeStack->raiseWidget(_itemToInterface[cur->itemAbove()]);
#else
      if(_volumeStackList->itemBelow(cur))
	_volumeStack->setCurrentWidget(_itemToInterface[_volumeStackList->itemBelow(cur)]);
      else if(_volumeStackList->itemAbove(cur))
	_volumeStack->setCurrentWidget(_itemToInterface[_volumeStackList->itemAbove(cur)]);
#endif

      VolumeInterface *vi = _itemToInterface[cur];
      delete vi;
      delete cur;
      _interfaceToItem.erase(vi);
      _itemToInterface.erase(cur);
    }

#if QT_VERSION < 0x040000
  if(_volumeStack->visibleWidget() == NULL)
    {
      _volumeViewerPage->close();
#ifdef VOLUMEGRIDROVER
      _volumeGridRover->setVolume(VolMagick::VolumeFileInfo());
#endif
    }
#else
  if(_volumeStack->currentWidget() == NULL)
    _volumeViewerPage->close();
#endif
}


void VolumeRoverMain::bilateralFilterSlot() {

  BilateralFilterDialog dialog(this);
  
  dialog.m_RadSigEdit->setValidator(new QDoubleValidator(this));
  dialog.m_SpatSigEdit->setValidator(new QDoubleValidator(this));
  dialog.m_FilRadEdit->setValidator(new QDoubleValidator(this));
  
  if(dialog.exec() == QDialog::Accepted) {
    QMessageBox::warning(this, tr("Bilateral Filter"),
			 tr("This feature has not been updated."), 0,0);
  }
  // old code... should be updated to match the new datamanager...
  /*
  if (m_SourceManager.hasSource()) {
    if (!m_RGBARendering) {
      BilateralFilterDialog dialog(this);

      dialog.m_RadSigEdit->setValidator(new QDoubleValidator(this));
      dialog.m_SpatSigEdit->setValidator(new QDoubleValidator(this));
      dialog.m_FilRadEdit->setValidator(new QDoubleValidator(this));

      if(dialog.exec() == QDialog::Accepted) {

	
	  //Check if we are doing in place filtering of the actual volume data file
	  //instead of simply the current subvolume buffer
	  //TODO: need to implement out-of-core filtering using VolMagick
	
	if(!dialog.m_Preview->isChecked())
	  {
	    if(QMessageBox::warning(this,
				    "Bilateral Filter",
				    "Are you sure you want to do this?\n"
				    "This operation will change the current loaded volume file\n"
				    "and cannot be un-done!",
				    QMessageBox::Cancel | QMessageBox::Default | QMessageBox::Escape,
				    QMessageBox::Ok) != QMessageBox::Ok)
	      return;
				 

	    VolMagick::Volume vol;
	    VolMagick::readVolumeFile(vol,
				      m_VolumeFileInfo.filename(),
				      getVarNum(),
				      getTimeStep());
	    vol.bilateralFilter(dialog.m_RadSigEdit->text().toDouble(),
				dialog.m_SpatSigEdit->text().toDouble(),
				dialog.m_FilRadEdit->text().toDouble());
	    VolMagick::writeVolumeFile(vol,
				       m_VolumeFileInfo.filename(),
				       getVarNum(),
				       getTimeStep());
	    QString tmpp(m_VolumeFileInfo.filename().c_str());
	    openFile(tmpp);
	    return;
	  }

	//BilateralFilter filter;

	OOCBilateralFilter oocFilter;

	//bool result;
	VolumeBuffer* densityBuffer;
	unsigned int dim[3];
	int i;

	unsigned char *bPtr;

			
	// get the volume buffer
	densityBuffer = m_ZoomedInRenderable.getVolumeBufferManager()->getVolumeBuffer(getVarNum());
	bPtr = (unsigned char *)densityBuffer->getBuffer();

	dim[0] = densityBuffer->getWidth();
	dim[1] = densityBuffer->getHeight();
	dim[2] = densityBuffer->getDepth();
		
	// apply the filter
	//result = filter.applyFilter(this, (unsigned char*)densityBuffer->getBuffer(), dim[0], dim[1], dim[2]);

	
	// set the filtered flag (for filtered subvolume saving)
	m_SubVolumeIsFiltered = true;
			
	Q3ProgressDialog progressDialog("Performing a bilateral filtering of the sub-volume.", "Cancel", dim[2], this, "Bilateral Filter", true);
	progressDialog.setProgress(0);

	// init
	oocFilter.setDataType(OutOfCoreFilter::U_CHAR);
	//oocFilter.initFilter(dim[0],dim[1],dim[2], 0.0, 255.0);
	oocFilter.initFilter(dim[0],dim[1],dim[2], 0.0, 255.0, atof(dialog.m_RadSigEdit->text().ascii()),
			     atof(dialog.m_SpatSigEdit->text().ascii()), atoi(dialog.m_FilRadEdit->text().ascii()));
	// pre-init the cache
	for (i=0; i < oocFilter.getNumCacheSlices(); i++)
	  oocFilter.addSlice((void *)(bPtr + i*dim[0]*dim[1]));
	// filter
	for (i=0; i < (int)dim[2] && m_SubVolumeIsFiltered; i++) {
	  oocFilter.filterSlice((void *)(bPtr + i*dim[0]*dim[1]));
	  if (i+oocFilter.getNumCacheSlices() < (int)dim[2])
	    oocFilter.addSlice((void *)(bPtr + (i+oocFilter.getNumCacheSlices())*dim[0]*dim[1]));

	  // update the progress and check for cancelation
	  progressDialog.setProgress(i);
	  qApp->processEvents();
	  if (progressDialog.wasCancelled()) {
	    // the filtering did not complete (this will also end the loop)
	    m_SubVolumeIsFiltered = false;
	    // let the buffer manager know what happened
	    m_ZoomedInRenderable.getVolumeBufferManager()->markBufferAsInvalid(getVarNum());
	  }
	}
	
	if (m_SubVolumeIsFiltered) {
	  // update
	  m_ZoomedIn->makeCurrent();
	  updateRoverRenderable(&m_ZoomedInRenderable, &m_ZoomedInExtents);
	printf("NewVolumeMainWindow::bilateralFilterSlot()\n");
	  m_ZoomedIn->updateGL();
	}
      }
    }

    else {
      QMessageBox::warning(this, tr("Bilateral Filter"),
			   tr("This feature is not available for RGBA volumes."), 0,0);
    }
  }
  else {
    QMessageBox::warning(this, tr("Bilateral Filter"),
			 tr("A volume must be loaded for this feature to work."), 0,0);
  }
  */

}

void VolumeRoverMain::addVolumeToStack(const VolMagick::VolumeFileInfo &vfi, bool force)
{
#if QT_VERSION < 0x040000
  QListViewItem *volitem = _volumeStackList->findItem(vfi.filename(),1);
#else
  QList<QTreeWidgetItem*> volitems = 
    _volumeStackList->findItems(QString(vfi.filename().c_str()),
                                //Consider making case insensitive for OSes with
                                //case insensitive filesystems.
                                Qt::MatchFixedString | Qt::MatchCaseSensitive,
                                1);
  QTreeWidgetItem* volitem = !volitems.isEmpty() ? volitems.first() : NULL;
#endif
  if(volitem)
    {
      if(!force &&
	 QMessageBox::information(this,
				  "File already opened",
				  QString("Close and re-open file %1?").arg(QFileInfo(QString(vfi.filename().c_str())).fileName()),
				  QMessageBox::Cancel,
				  QMessageBox::Ok) == QMessageBox::Cancel)
	    return;


      delete _itemToInterface[volitem];
      delete volitem;
      _interfaceToItem.erase(_itemToInterface[volitem]);
      _itemToInterface.erase(volitem);
    }

  VolumeInterface *vi = new VolumeInterface(vfi,_ui->_mainFrame);
#if QT_VERSION < 0x040000
  volitem = new QListViewItem(_volumeStackList,
                              QFileInfo(vfi.filename()).fileName(),
                              vfi.filename());
#else
  volitem = new QTreeWidgetItem(_volumeStackList);
  volitem->setText(0,QFileInfo(vfi.filename().c_str()).fileName());
  volitem->setText(1,vfi.filename().c_str());
#endif
  _itemToInterface[volitem] = vi;
  _interfaceToItem[vi] = volitem;
  _volumeStack->addWidget(vi);
#if QT_VERSION < 0x040000
  _volumeStack->raiseWidget(vi);
#else
  _volumeStack->setCurrentWidget(vi);
#endif
  _volumeViewerPage->openVolume(vfi);

#ifdef VOLUMEGRIDROVER
  _volumeGridRover->setVolume(vfi);
#endif

  //The following connection should do what the user expects, reloading the
  //just modified volume.  Since the volume inteface of the modified volume
  //should already be open (so the user can actually do the modification),
  //the volume viewers should also have the same volume currently visible.
  //Thus it makes sense to reload the last modified volume originating from
  //any VolumeInterface without checking anything...
  connect(vi,
	  SIGNAL(volumeModified(const VolMagick::VolumeFileInfo&)),
	  _volumeViewerPage,
	  SLOT(openVolume(const VolMagick::VolumeFileInfo&)));
}

void VolumeRoverMain::raiseSelected()
{
#if QT_VERSION < 0x040000
  QListViewItemIterator it(_volumeStackList, QListViewItemIterator::Selected);
  _volumeStack->raiseWidget(_itemToInterface[it.current()]);
#else
  QTreeWidgetItemIterator it(_volumeStackList, QTreeWidgetItemIterator::Selected);
  _volumeStack->setCurrentWidget(_itemToInterface[*it]);
#endif
}

void VolumeRoverMain::loadShownVolume(int id)
{
  QWidget *vi = _volumeStack->widget(id);
  if(vi)
    {
      _volumeViewerPage->openVolume(static_cast<VolumeInterface*>(vi)->volumeFileInfo());
#ifdef VOLUMEGRIDROVER
      _volumeGridRover->setVolume(static_cast<VolumeInterface*>(vi)->volumeFileInfo());
#endif
    }
}

#ifdef VOLUMEGRIDROVER
void VolumeRoverMain::syncIsocontourValuesWithVolumeGridRover()
{
  
}
#endif


void VolumeRoverMain::addObject(const std::string& name,
                                const boost::any& obj)
{
  _objects[name]=obj;
}

void VolumeRoverMain::removeObject(const std::string& name)
{
  _objects.erase(name);
}

void VolumeRoverMain::clearObjects()
{
  _objects.clear();
}

boost::any VolumeRoverMain::getObject(const std::string& name)
{
  //Avoid adding to the map if we don't actually have an object by that name
  if(hasObject(name))
    return _objects[name];
  else return boost::any();
}

bool VolumeRoverMain::hasObject(const std::string& name) const
{
  if(_objects.find(name) == _objects.end())
    return false;
  return true;
}

bool VolumeRoverMain::addFile(const std::string& filename,
                              const std::string& objname)
{
  bool ok = true;
  std::string name = !objname.empty() ? objname : filename;

  //Try loading as a volume
  if(!ok)
    {
      try
        {
          VolMagick::VolumeFileInfo vfi(filename);
          addObject(name, vfi);
        }
      catch(std::exception& e)
        {
          ok = false;
        }
    }

  //Try loading as geometry
  if(!ok)
    {
      try
        {
          //Use geometry_t for read because read doesn't yet
          //support cvcgeom_t directly.
          cvcraw_geometry::geometry_t geom;
          cvcraw_geometry::read(geom,filename);
          cvcraw_geometry::cvcgeom_t cvc_geom = geom;
          addObject(name, cvc_geom);
        }
      catch(std::exception& e)
        {
          ok = false;
        }
    }

  return ok;
}

boost::any VolumeRoverMain::selectObject(const std::string& name)
{
  _selectedObject = _objects.find(name);
  return getSelectedObject();
}

boost::any VolumeRoverMain::getSelectedObject()
{
  if(_selectedObject == _objects.end())
    return boost::any();
  else
    return _selectedObject->second;
}
