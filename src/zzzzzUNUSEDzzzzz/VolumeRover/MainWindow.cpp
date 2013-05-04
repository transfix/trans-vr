#include <ui_MainWindowBase.h>

#include <VolumeRover/MainWindow.h>
#include <VolumeRover2/VolumeViewerPage.h>

#include <VolumeFileTypes/SourceManager.h>
#include <VolumeFileTypes/VolumeFileSource.h>

#include <VolumeRover/optionsdialogimpl.h>

#include <iostream>

#include <QString>


using namespace std;


MainWindow::MainWindow(QWidget* parent, const char* name, Qt::WFlags f)
	: QMainWindow(parent, f), _ui(NULL)
{

  resize(920, 720);
  setMinimumSize(QSize(900, 720));

  //set mainwindow ui files
  _centralWidget = new QWidget;
  setCentralWidget(_centralWidget);
  _ui = new Ui::MainWindowBase;
  _ui->setupUi(_centralWidget);
  //******************************	

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



  initializeMainWindowMenubar();
  initializeFileMenu();
}

MainWindow::~MainWindow()
{

}
void MainWindow::initializeMainWindowMenubar()
{

  // File menu
  menuFile = menuBar()->addMenu(tr("&File"));
  m_Action_FileOpen = new QAction(this);
  m_Action_FileOpen->setText("Open");
  menuFile->addAction(m_Action_FileOpen);

  m_Action_FileSave = new QAction(this);
  m_Action_FileSave->setText("Save");
  menuFile->addAction(m_Action_FileSave);

  m_Action_FileOptions = new QAction(this);
  m_Action_FileOptions->setText("Options");
  menuFile->addAction(m_Action_FileOptions);

  m_Action_FileExit = new QAction(this);
  m_Action_FileExit->setText("Exit");
  menuFile->addAction(m_Action_FileExit);

  menuFile = menuBar()->addMenu(tr("&View"));

  // menuFile = menuBar()->addMenu(tr("&Geometry"));
  menuFile = menuBar()->addMenu(tr("&Servers"));
  menuFile = menuBar()->addMenu(tr("&Animation"));
  menuFile = menuBar()->addMenu(tr("&Tools"));
  menuFile = menuBar()->addMenu(tr("&Help"));



    /*
    m_Action_RecentFileListMenu = new QMenu(menuFile);
   
    menuView = menuBar()->addMenu(tr("&View"));
   

    menuAnimation = menuBar()->addMenu(tr("&Animation"));
    menuOptions = menuBar()->addMenu(tr("&Options"));
    m_Menu_LightingPopup = menuBar()->addMenu(tr("&Lighting"));
    menuUtilities = menuBar()->addMenu(tr("&Utilities"));
    
    menuHelp = menuBar()->addMenu(tr("&Help"));
    menuHelp->setTearOffEnabled(true);

    menuBar()->addAction(menuFile->menuAction());
    menuBar()->addAction(menuView->menuAction());
    menuBar()->addAction(menuAnimation->menuAction());
    menuBar()->addAction(menuOptions->menuAction());
    menuBar()->addAction(m_Menu_LightingPopup->menuAction());
    menuBar()->addAction(menuUtilities->menuAction());
    menuBar()->addAction(menuHelp->menuAction());

    // Create Actions
    m_Action_FileOpen = new QAction(this);
    m_Action_FileOpen->setText("Open");

    m_Action_FileOpenMulti = new QAction(this);
    m_Action_FileOpenMulti->setText("Open Multi");

    m_Action_DownloadPDB = new QAction(this);
    m_Action_DownloadPDB->setText("DownloadPDB");

    m_Action_SaveImage = new QAction(this);
    m_Action_SaveImage->setText("Save Render Image");

    m_Action_RecentFileListDummy = new QAction(this);
    m_Action_RecentFileListDummy->setText("No File");
    m_Action_RecentFileListDummy->setEnabled(false);
    
    m_Action_RecentFileListMenu = menuFile->addMenu(tr("&Recent Files"));
    m_Action_RecentFileListMenu->addAction(m_Action_RecentFileListDummy);

    m_Action_Exit = new QAction(this);
    m_Action_Exit->setText("Exit");

    menuFile->addAction(m_Action_FileOpen);
    menuFile->addAction(m_Action_FileOpenMulti);
    menuFile->addAction(m_Action_DownloadPDB);
    menuFile->addAction(m_Action_SaveImage);
    menuFile->addAction(m_Action_RecentFileListMenu->menuAction());
    menuFile->addSeparator();
    menuFile->addAction(m_Action_Exit);
    
    m_Action_OrthographicView = new QAction(this);
    m_Action_OrthographicView->setText("Orthographic");
    m_Action_OrthographicView->setCheckable(true);
    m_Action_PerspectiveView = new QAction(this);
    m_Action_PerspectiveView->setText("Perspective");
    m_Action_PerspectiveView->setCheckable(true);
    m_Action_PerspectiveView->setChecked(true);
    m_Action_SetViewParameters = new QAction(this);

    menuProjection = menuView->addMenu(tr("&Projection"));
    menuProjection->addAction(m_Action_OrthographicView);
    menuProjection->addAction(m_Action_PerspectiveView);

    m_Action_SetViewParameters->setText("Set View Parameters");
    m_Action_SplitView = new QAction(this);
    m_Action_SplitView->setText("Split View");
    m_Action_SyncView = new QAction(this);
    m_Action_SyncView->setText("Sync View");
    m_Action_SyncView->setCheckable(true);
    m_Action_FullScreen = new QAction(this);
    m_Action_FullScreen->setText("Full Screen");
    m_Action_Stereo = new QAction(this);
    m_Action_Stereo->setText("Stereo");
    m_Action_Stereo->setCheckable(true);
    m_Action_ForceMeshRender = new QAction(this);
    m_Action_ForceMeshRender->setText("Force Mesh Render");
    m_Action_ForceMeshRender->setCheckable(true);
    
   
    menuView->addAction(menuProjection->menuAction());
    menuView->addAction(m_Action_SetViewParameters);
    menuView->addSeparator();
    menuView->addAction(m_Action_SplitView);
    menuView->addAction(m_Action_SyncView);
    menuView->addSeparator();
    menuView->addAction(m_Action_FullScreen);
    menuView->addAction(m_Action_Stereo);
    menuView->addAction(m_Action_ForceMeshRender);

    m_Action_StartRecording = new QAction(this);
    m_Action_StartRecording->setText("Start Recording");
    m_Action_StopRecording = new QAction(this);
    m_Action_StopRecording->setText("Stop Recording");
    m_Action_PlaybackAnimation = new QAction(this);
    m_Action_PlaybackAnimation->setText("Playback Animation");
    m_Action_RecordAnimation = new QAction(this);
    m_Action_RecordAnimation->setText("Record Animation");
    m_Action_PlaybackMovie = new QAction(this);
    m_Action_PlaybackMovie->setText("Playback Movie");
    m_Action_RayTrace = new QAction(this);
    m_Action_RayTrace->setText("Ray Trace");
    m_Action_RayTrace->setCheckable(true);
    m_Action_BackgroundColor = new QAction(this);
    m_Action_BackgroundColor->setText("Background Color");
    m_Action_DisplayGrid = new QAction(this);
    m_Action_DisplayGrid->setText("Display Grid");
    m_Action_DisplayGrid->setCheckable(true);
    m_Action_DisplayGrid->setChecked(true);
    m_Action_MouseKeyboardFunc = new QAction(this);
    m_Action_MouseKeyboardFunc->setText("Mouse-Keyboard Function");
    m_Action_TransformObject = new QAction(this);
    m_Action_TransformObject->setText("Transform Object");
    m_Action_TransformObject->setEnabled(false);
    m_Action_SelectObjects = new QAction(this);
    m_Action_SelectObjects->setText("Select Objects");
    m_Action_SelectObjects->setCheckable(true);
    m_Action_GlobalBoundingBox = new QAction(this);
    m_Action_GlobalBoundingBox->setText("Global Bounding Box");
    m_Action_GlobalBoundingBox->setCheckable(true);
    m_Action_GlobalBoundingBox->setChecked(false);
    m_Action_DataBoundingBox = new QAction(this);
    m_Action_DataBoundingBox->setText("Data Bounding Box");
    m_Action_DataBoundingBox->setCheckable(true);
    m_Action_ShowViewInformation = new QAction(this);
    m_Action_ShowViewInformation->setText("Show View Information");
    m_Action_ShowViewInformation->setCheckable(true);
    m_Action_ShowViewInformation->setChecked(true);
    m_Action_ShowMousePosition = new QAction(this);
    m_Action_ShowMousePosition->setText("Show Mouse Position");
    m_Action_ShowMousePosition->setCheckable(true);
    m_Action_ShowMousePosition->setChecked(true);

    m_Action_Script = new QAction(this);
    m_Action_Script->setText("Script");
    m_Action_ConstructSurface = new QAction(this);
    m_Action_ConstructSurface->setText("Construct Surface");
    m_Action_ConstructNURBSSurface = new QAction(this);
    m_Action_ConstructNURBSSurface->setText("Construct NURBS Surface");
    m_Action_SurfaceAreaAndVolume = new QAction(this);
    m_Action_SurfaceAreaAndVolume->setText("Surface Area and Volume");
    m_Action_ConstructVolume = new QAction(this);
    m_Action_ConstructVolume->setText("Construct Volume");
    m_Action_ConstructDepthColoredVolume = new QAction(this);
    m_Action_ConstructDepthColoredVolume->setText("Construct Depth-Colored Volume");
    m_Action_ConstructPockets = new QAction(this);
    m_Action_ConstructPockets->setText("Construct Pockets");
    m_Action_ConstructPocketTunnelStableManifold = new QAction(this);
    m_Action_ConstructPocketTunnelStableManifold->setText("Construct Pocket-Tunnel by Stable Manifold");
    m_Action_ConstructHLSPockets = new QAction(this);
    m_Action_ConstructHLSPockets->setText("Construct HLS Pockets");
    m_Action_GetCurvature = new QAction(this);
    m_Action_GetCurvature->setText("Get Curvatures");
    m_Action_FormMatch = new QAction(this);
    m_Action_FormMatch->setText("Form a (MACT) Match");
    m_Action_ComputeEnergyGB = new QAction(this);
    m_Action_ComputeEnergyGB->setText("Compute Energy (GB)");
    m_Action_ComputeForceFieldGB = new QAction(this);
    m_Action_ComputeForceFieldGB->setText("Compute Force Field (GB)");
    m_Action_ComputeEnergyPotentialPB = new QAction(this);
    m_Action_ComputeEnergyPotentialPB->setText("Compute Energy+Potential (PB)");
    m_Action_Get2DSlice = new QAction(this);
    m_Action_Get2DSlice->setText("Get2DSlice");
    m_Action_ElucidateSecondaryStructure = new QAction(this);
    m_Action_ElucidateSecondaryStructure->setText("Elucidate Secondary Structures");
    m_Action_F2Dock = new QAction(this);
    m_Action_F2Dock->setText("F2Dock");

    m_Action_Contents = new QAction(this);
    m_Action_Contents->setText("Contents");
    m_Action_Index = new QAction(this);
    m_Action_Index->setText("Index");
    m_Action_Acknowledgements = new QAction(this);
    m_Action_Acknowledgements->setText("Acknowledgements");

    menuAnimation->addAction(m_Action_StartRecording);
    menuAnimation->addAction(m_Action_StopRecording);
    menuAnimation->addAction(m_Action_PlaybackAnimation);
    menuAnimation->addSeparator();
    menuAnimation->addAction(m_Action_RecordAnimation);
    menuAnimation->addSeparator();
    menuAnimation->addAction(m_Action_PlaybackMovie);
    menuOptions->addAction(m_Action_RayTrace);
    menuOptions->addAction(m_Action_BackgroundColor);
    menuOptions->addAction(m_Action_DisplayGrid);
    menuOptions->addSeparator();
    menuOptions->addAction(m_Action_MouseKeyboardFunc);
    menuOptions->addAction(m_Action_TransformObject);
    menuOptions->addAction(m_Action_SelectObjects);
    menuOptions->addSeparator();
    menuOptions->addAction(m_Action_GlobalBoundingBox);
    menuOptions->addAction(m_Action_DataBoundingBox);
    menuOptions->addSeparator();
    menuOptions->addAction(m_Action_ShowViewInformation);
    menuOptions->addAction(m_Action_ShowMousePosition);
    menuUtilities->addAction(m_Action_Script);
    menuUtilities->addSeparator();
    menuUtilities->addAction(m_Action_ConstructSurface);
    menuUtilities->addAction(m_Action_ConstructNURBSSurface);
    menuUtilities->addAction(m_Action_SurfaceAreaAndVolume);
    menuUtilities->addSeparator();
    menuUtilities->addAction(m_Action_ConstructVolume);
    menuUtilities->addAction(m_Action_ConstructDepthColoredVolume);
    menuUtilities->addSeparator();
    menuUtilities->addAction(m_Action_ConstructPockets);
    menuUtilities->addAction(m_Action_ConstructPocketTunnelStableManifold);
    menuUtilities->addAction(m_Action_ConstructHLSPockets);
    menuUtilities->addSeparator();
    menuUtilities->addAction(m_Action_GetCurvature);
    menuUtilities->addAction(m_Action_FormMatch);
    menuUtilities->addAction(m_Action_ComputeEnergyGB);
    menuUtilities->addAction(m_Action_ComputeForceFieldGB);
    menuUtilities->addAction(m_Action_ComputeEnergyPotentialPB);
    menuUtilities->addSeparator();
    menuUtilities->addAction(m_Action_Get2DSlice);
    menuUtilities->addAction(m_Action_ElucidateSecondaryStructure);
    menuUtilities->addAction(m_Action_F2Dock);
    menuHelp->addAction(m_Action_Contents);
    menuHelp->addAction(m_Action_Index);
    menuHelp->addSeparator();
    menuHelp->addAction(m_Action_Acknowledgements);
    */

}


void MainWindow::initializeFileMenu()
{
  connect(m_Action_FileOpen, SIGNAL(triggered()), SLOT(fileOpenSlot()));
  //connect(m_Action_FileSave, SIGNAL(triggered()), SLOT(fileSaveSlot()));
  connect(m_Action_FileSave, SIGNAL(triggered()), SLOT(notYetImplementedSlot()));
  connect(m_Action_FileOptions, SIGNAL(triggered()), SLOT(fileOptionsSlot()));
  connect(m_Action_FileExit, SIGNAL(triggered()), SLOT(fileExitSlot()));
}


bool MainWindow::notYetImplementedSlot()
{
  cout << "ERROR: this feature is not yet implemented." << endl;
  QMessageBox::critical( this, "Error", "This feature is not yet imlpemented." );


}

bool MainWindow::fileOpenSlot()
{
  QStringList filenames = QFileDialog::getOpenFileNames(this,
                                                        "Open File",
                                                        QString::null,
                                                        "RawIV (*.rawiv);;"
							"RawV (*.rawv);;"
							"MRC (*.mrc);;"
							"cvc-raw geometry (*.raw *.rawn *.rawc *.rawnc);;"
							"All Files (*)");

  for(QStringList::iterator it = filenames.begin();
      it != filenames.end();
      ++it)
    {
      if((*it).isEmpty()) continue;
      
      std::string cur((*it).toAscii());
      
      // FIXME: check if this is a .rawiv file


      
      _volumeViewerPage->openVolume(VolMagick::VolumeFileInfo(cur));


    }



  /*
  bool addedNewDataSet = true;
  QString filename;
  //fileNames = Q3FileDialog::getOpenFileNames(getFilter(m_DataManager->getDataTypes(), m_DataManager->getNumberOfDataTypes()), "../DataSet", this);
  filename = QFileDialog::getOpenFileName(this,
					  "Select one or more files to open",
					  "",
					  ".rawiv");
  
  qDebug("Opening: %s",filename.ascii());
  SourceManager m_SourceManager;
  QString m_CurrentVolumeFilename = filename;
  VolumeFileSource* source = new VolumeFileSource(filename, "/workspace/arand/tmp");
  if (source) {
    if (!source->open(this)) {
      QMessageBox::critical( this, "Error opening the file", 
			     "An error occured while attempting to open the file: \n"+source->errorReason() );
      delete source;
    }
    else {
      
      //updateRecentlyUsedList(filename);
      m_SourceManager.setSource(source);
      if(!m_ZoomedInRenderable.getVolumeBufferManager()->setSourceManager(&m_SourceManager))
	{
	  QMessageBox::critical(this, "Error opening the file", 
				"An error occured while attempting to open the file: Error setting source manager");
	  delete source;
	  return;
	}

    }
  }
  */

}

bool MainWindow::fileSaveSlot()
{

}

bool MainWindow::fileOptionsSlot()
{

  cout << "Options slot" << endl;

  OptionsDialog dialog(this, 0, true);

  dialog.exec();

  QMessageBox::critical( this, "Warning", "Options dialog currently has no effects.");
  /*
  switch (m_UpdateMethod) {
  case UMInteractive:
    dialog.m_Interactive->setChecked(true);
    break;
  case UMDelayed:
    dialog.m_Delayed->setChecked(true);
    break;
  case UMManual:
    dialog.m_Manual->setChecked(true);
    break;
  };
  
  if (m_RGBARendering) {
    dialog.m_RGBA->setChecked(true);
  }
  else {
    dialog.m_Single->setChecked(true);
  }
  
  if (m_ThumbnailRenderable.getShadedVolumeRendering()) {
    dialog.m_Shaded->setChecked(true);
  }
  else {
    dialog.m_Unshaded->setChecked(true);
  }
  
  // disable if shaded rendering is unavailable
  if (m_ThumbnailRenderable.getVolumeRenderer()->isShadedRenderingAvailable() == false) {
    printf("shaded render is not available\n");
    dialog.m_Shaded->setEnabled(false);
    dialog.m_Unshaded->setEnabled(false);
  }
  else
    printf("shaded render is available\n");
  
  switch (m_TransferFunc) {
  case TF1D:
    dialog.m_1DTransferFunc->setChecked(true);
    break;
  case TF2D:
    dialog.m_2DTransferFunc->setChecked(true);
    break;
  case TF3D:
    dialog.m_3DTransferFunc->setChecked(true);
    break;
  }
  
  dialog.m_ZoomedInSurface->setChecked(m_ZoomedInRenderable.getShowIsosurface());
  dialog.m_ZoomedOutSurface->setChecked(m_ThumbnailRenderable.getShowIsosurface());
  
  QDir dir(getCacheDir());
  dialog.m_CacheDir->setText(dir.absPath());
  dialog.setColor(m_ZoomedIn->getBackgroundColor());
  
  if (dialog.exec() == QDialog::Accepted) {
    if (dialog.m_Interactive->isChecked()) {
      m_UpdateMethod = UMInteractive;
    }
    else if (dialog.m_Delayed->isChecked()) {
      m_UpdateMethod = UMDelayed;
    }
    else if (dialog.m_Manual->isChecked()) {
      m_UpdateMethod = UMManual;
    }
    m_ZoomedInRenderable.setShowIsosurface(dialog.m_ZoomedInSurface->isChecked());
    m_ThumbnailRenderable.setShowIsosurface(dialog.m_ZoomedOutSurface->isChecked());
    
    // just set flags for shaded rendering
    // updating of volumes and other render state is done
    // implicitly in the RGBA/Single switch below
    if (dialog.m_Shaded->isChecked()) {
      m_ThumbnailRenderable.setShadedVolumeRendering(true);
      m_ZoomedInRenderable.setShadedVolumeRendering(true);
      
      m_ThumbnailRenderable.getVolumeRenderer()->enableShadedRendering();
      m_ZoomedInRenderable.getVolumeRenderer()->enableShadedRendering();
    }
    else if (dialog.m_Unshaded->isChecked()) {
      m_ThumbnailRenderable.setShadedVolumeRendering(false);
      m_ZoomedInRenderable.setShadedVolumeRendering(false);
      
      m_ThumbnailRenderable.getVolumeRenderer()->disableShadedRendering();
      m_ZoomedInRenderable.getVolumeRenderer()->disableShadedRendering();
    }
    
    if (dialog.m_RGBA->isChecked()) {
      m_RGBARendering = true;
      updateVariableInfo(false);
      getThumbnail();
      explorerChangedSlot();
    }
    else if (dialog.m_Single->isChecked()) {
      m_RGBARendering = false;
      updateVariableInfo(false);
      getThumbnail();
      explorerChangedSlot();
    }
    
    if (dialog.m_1DTransferFunc->isChecked()) {
      m_TransferFunc = TF1D;
      m_ColorToolbarStack->raiseWidget(m_ColorTable);
    } else if (dialog.m_2DTransferFunc->isChecked()) {
      //m_TransferFunc = TF2D;
      //m_ColorToolbarStack->raiseWidget(m_ColorTable2D);
    } else if (dialog.m_3DTransferFunc->isChecked()) {
      m_TransferFunc = TF3D;
      QMessageBox::warning(this, "Warning", "3D Transfer function is not yet implemented.");
    }
    
    QDir result(dialog.m_CacheDir->text());
    QDir resultBase(result);
    
    // prepare the cache dir
    if (!result.exists("VolumeCache")) {
      result.mkdir("VolumeCache");
      result.cd("VolumeCache");
    }
    else {
      result.cd("VolumeCache");
    }
    
    if (m_CacheDir!=result) {
      int button = QMessageBox::warning(this, "Warning", "Changing the cache directory will unload any"
					" data currently loaded.", "Continue", "Cancel");
      if (button==0) {
	QSettings settings;
	settings.insertSearchPath(QSettings::Windows, "/CCV");
	settings.writeEntry("/Volume Rover/CacheDir", resultBase.absPath());
	
	m_CacheDir = result;
	m_SourceManager.resetSource();
	m_ZoomedInRenderable.setShowVolumeRendering(false);
	m_ThumbnailRenderable.setShowVolumeRendering(false);
      }
      
      qDebug("Change dir");
    }
    
    // prepare the background color
    QColor color = dialog.getColor();
    setSavedColor(dialog.getColor());
    m_ZoomedIn->setBackgroundColor(dialog.getColor());
    m_ZoomedInRenderable.setDepthCueColor(Qt::red/255.0,
					  Qt::green/255.0, Qt::blue/255.0);
    m_ZoomedInRenderable.setWireCubeColor(1.0-Qt::red,
					  1.0-Qt::green, 1.0-Qt::blue);
    m_ZoomedIn->updateGL();
    m_ZoomedOut->setBackgroundColor(dialog.getColor());
    m_ThumbnailRenderable.setDepthCueColor(Qt::red/255.0,
					   Qt::green/255.0,Qt::blue/255.0);
    m_ThumbnailRenderable.setRover3DWidgetColor(1.0-Qt::red,
						1.0-Qt::green, 1.0-Qt::blue);
    m_ZoomedOut->updateGL();
  }
  */
  
}

bool MainWindow::fileExitSlot()
{

  // FUTURE: do any necessary cleanup
  //         warn the user?

  close();
}
