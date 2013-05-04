/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: Anthony Thane <thanea@ices.utexas.edu>
	         John Wiggins <prok@ices.utexas.edu>
		 Jose Rivera <transfix@ices.utexas.edu>
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

/* $Id: newvolumemainwindow.cpp 3513 2011-01-27 16:08:53Z arand $ */

#include <VolumeRover/newvolumemainwindow.h>

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include "time.h"
//#include <OpenGLViewer.h>
#include <ColorTable/ColorTable.h>
#include <ColorTable2D/ColorTable2D.h>
#include <qmessagebox.h>
//Added by qt3to4:
#include <QTimerEvent>
#include <QCustomEvent>
#include <QKeyEvent>
#include <QEvent>
#include <QPixmap>
#include <QColor>
//#include "RawIVSource.h"
#include <VolumeFileTypes/VolumeFileSource.h>
#include <VolumeFileTypes/DataCutterSource.h>
#include <VolumeFileTypes/VolumeFileSink.h>
#include <VolumeFileTypes/VolumeTranscriber.h>
#include <VolumeRover/filelistdialogimpl.h>
#include <VolumeRover/optionsdialogimpl.h>
#include <VolumeRover/serverselectordialogimpl.h>
#include <VolumeRover/imagesavedialogimpl.h>
#include "VolumeRover/volumesavedialog.Qt3.h"
#include <VolumeRover/terminal.h>
#include "VolumeRover/bilateralfilterdialog.Qt3.h"
#include "VolumeRover/gdtvfilterdialog.Qt3.h"
#include "VolumeRover/highlevelsetrecondialog.Qt3.h"
#include "VolumeRover/contrastenhancementdialog.Qt3.h"
#include "VolumeRover/anisotropicdiffusiondialog.Qt3.h"
#include "VolumeRover/pedetectiondialog.Qt3.h"
#include "VolumeRover/slicerenderingdialog.Qt3.h"
#include <VolumeRover/BoundaryPointCloudDialog.h>

#include <VolumeRover/ImageViewer.h>
#include <VolumeFileTypes/DownLoadManager.h>
#include <ByteOrder/ByteSwapping.h>
#include <VolumeRover/VolMagickEventHandler.h>
#include <qapplication.h>

#include <q3listbox.h>
#include <q3filedialog.h>
#include <qfileinfo.h>
#include <qinputdialog.h>
#include <qradiobutton.h>
#include <qcheckbox.h>
#include <q3frame.h>
#include <qfile.h>
#include <qstringlist.h>
#include <q3textstream.h>
#include <q3textedit.h>
#include <qaction.h>
#include <qcombobox.h>
#include <qspinbox.h>
#include <qimage.h>
#include <qslider.h>
#include <q3widgetstack.h>
#include <qtabwidget.h>
#include <qvalidator.h>
#include <qthread.h>
#include <q3buttongroup.h>
#include <qmap.h>
#include <q3progressbar.h>
#include <qstatusbar.h>
//#include <BilateralFilter.h>

#include <Filters/OOCBilateralFilter.h>
#include <Filters/ContrastEnhancement.h>

#include <VolumeWidget/Extents.h>
#include <GeometryFileTypes/GeometryLoader.h>
#include <qsettings.h>
#include <VolumeWidget/TrackballRotateInteractor.h>
#include <VolumeWidget/ScaleInteractor.h>

#include <AnimationMaker/Animation.h>
#include <VolumeRover/segmentationdialogimpl.h>

#include "VolumeRover/convertisosurfacetogeometrydialogbase.Qt3.h" //use the base dialog for now

#include <VolumeLibrary/CG_RenderDef.h>

//#include <Segmentation.h>

// contour tree
#include <contourtree/computeCT.h>

// WorkerThreads
//#include "WorkerThread.h"

//#include "IPolyRenderable.h"
//#include "../ipoly/src/ipoly.h"
//#include "../ipoly/src/ipolyutil.h"

#include <RenderServers/RenderServer.h>

#include <XmlRPC/XmlRpc.h>
#include <Segmentation/SegCapsid/segcapsid.h>
#include <Segmentation/SegMonomer/segmonomer.h>
#include <Segmentation/SegSubunit/segsubunit.h>
#include <Segmentation/SecStruct/secstruct.h>

#include <Filters/Smoothing.h>
#include <VolumeRover/SmoothingDialog.h>

// Only include corba Render servers if using CORBA
#ifdef USINGCORBA
#include <RenderServers/RaycastRenderServer.h>
#include <RenderServers/TextureRenderServer.h>
#include <RenderServers/IsocontourRenderServer.h>
#endif

#ifdef USING_PE_DETECTION
#include <PEDetection/VesselSeg.h>
#endif

#ifdef USING_POCKET_TUNNEL
#undef Min
#undef Max
#include <PocketTunnel/pocket_tunnel.h>
#endif

#ifdef VOLUMEGRIDROVER
#include <VolumeGridRover/SurfRecon.h>
#endif

#ifdef USING_TIGHT_COCONE
#include <TightCocone/tight_cocone.h>
#include <VolumeRover/TightCoconeDialog.h>
#endif

#ifdef USING_CURATION
#include <Curation/Curation.h>
#include <VolumeRover/CurationDialog.h>
#endif

//#ifdef USING_HLEVELSET
#include <HLevelSet/HLevelSet.h>
//#endif

#ifdef USING_SECONDARYSTRUCTURES
#include <SecondaryStructures/skel.h>
#include <Histogram/histogram.h>
using namespace SecondaryStructures;
#endif


#ifdef USING_SKELETONIZATION
#include <Skeletonization/Skeletonization.h>
#include <VolumeRover/SkeletonizationDialog.h>
#endif



#include <multi_sdf/multi_sdf.h>
#include <SignDistanceFunction/sdfLib.h>
#include <VolumeRover/SignedDistanceFunctionDialog.h>

//LBIE stuff
#include <VolumeRover/LBIEMeshingDialog.h>
#include <VolumeRover/LBIEQualityImprovementDialog.h>
#include <LBIE/LBIE_Mesher.h>
#include <LBIE/quality_improve.h>

#include <boost/any.hpp>
#include <boost/scoped_array.hpp>

#include <VolumeRover/projectgeometrydialog.h>
#include <Filters/project_verts.h>
#include <cvcraw_geometry/cvcraw_geometry.h>

#ifdef USING_MSLEVELSET
//MS Level Set dialog box
#include "mslevelsetdialog.h"
//Modified Mumford Shah segmentation
#include <MSLevelSet/levelset3D.h>
#endif

#ifdef USING_RECONSTRUCTION
#include <Reconstruction/B_spline.h>
#include <Reconstruction/utilities.h>
#include <Reconstruction/Reconstruction.h>
#include <VolumeRover/reconstructiondialogimpl.h>
#endif

using namespace std;
using namespace XmlRpc;

#warning TODO: why are these global vars here?? put them in the HLevelSet object where they belong!
float edgelength;
int   end,Max_dim;

/* Need these to signal the GUI thread the segmentation result so it can popup a message to the user */
class SegmentationFailedEvent : public QCustomEvent
{
public:
  SegmentationFailedEvent(const QString &m) : QCustomEvent(QEvent::User+100), msg(m) {}
  QString message() const { return msg; }
private:
  QString msg;
};

class SegmentationFinishedEvent : public QCustomEvent
{
public:
  SegmentationFinishedEvent(const QString &m) : QCustomEvent(QEvent::User+101), msg(m) {}
  QString message() const { return msg; }
private:
  QString msg;
};

#ifdef USING_PE_DETECTION
class PEDetectionFailedEvent : public QCustomEvent
{
public:
  PEDetectionFailedEvent(const QString& m) : QCustomEvent(QEvent::User+102), msg(m) {}
  QString message() const { return msg; }
private:
  QString msg;
};

class PEDetectionFinishedEvent : public QCustomEvent
{
public:
  PEDetectionFinishedEvent(const QString& m) : QCustomEvent(QEvent::User+103), msg(m) {}
  QString message() const { return msg; }
private:
  QString msg;
};
#endif

#ifdef USING_POCKET_TUNNEL
class PocketTunnelFailedEvent : public QCustomEvent
{
public:
  PocketTunnelFailedEvent(const QString& m) : QCustomEvent(QEvent::User+104), msg(m) {}
  QString message() const { return msg; }
private:
  QString msg;
};

class PocketTunnelFinishedEvent : public QCustomEvent
{
public:
  PocketTunnelFinishedEvent(const QString& m, Geometry *result) : 
    QCustomEvent(QEvent::User+105), msg(m), resultMesh(result) {}
  QString message() const { return msg; }

  Geometry *resultMesh;
private:
  QString msg;
};
#endif

class TilingInfoEvent : public QCustomEvent
{
public:
  TilingInfoEvent(const QString& m, const boost::shared_ptr<Geometry>& result) :
    QCustomEvent(QEvent::User+106), msg(m), resultMesh(result) {}
  QString message() const { return msg; }

  boost::shared_ptr<Geometry> resultMesh;
private:
  QString msg;
};

#ifdef USING_MSLEVELSET
class MSLevelSetFailedEvent : public QCustomEvent
{
public:
  MSLevelSetFailedEvent(const QString& m) : QCustomEvent(QEvent::User+107), msg(m) {}
  QString message() const { return msg; }
private:
  QString msg;
};

class MSLevelSetFinishedEvent : public QCustomEvent
{
public:
  MSLevelSetFinishedEvent(const QString& m,
			  const VolMagick::Volume& result,
			  const boost::shared_ptr<Geometry>& contourResult) :
    QCustomEvent(QEvent::User+108), msg(m), 
    resultVol(result), resultMesh(contourResult)
  {
    *resultVol; //force local copy
  }
  QString message() const { return msg; }

  VolMagick::Volume resultVol;
  boost::shared_ptr<Geometry> resultMesh;
private:
  QString msg;  
};
#endif

// the animation sampling interval (in milliseconds)
static const unsigned int c_SampleInterval = 500;

static inline unsigned char mapToChar(double val)
{
	int inval = (int)(val*255);
	inval = ( inval<255 ? inval : 255 );
	inval = ( inval>0 ? inval : 0);
	return (unsigned char) inval;
}

const char * const c_VolumeRoverVersionTag = "$Name: HEAD $";

NewVolumeMainWindow::NewVolumeMainWindow( QWidget* parent, const char* name, Qt::WFlags f )
	: NewVolumeMainWindowBase( parent, name, f ),
	m_RecentFiles(m_FileMenu, this, SLOT(recentFileSlot(int))),
	m_ZoomedInRenderable(&m_ZoomedInExtents, &m_Geometries), 
	m_ZoomedInExtents(0.25, 0.75, 0.25, 0.75, 0.25, 0.75),
	m_ThumbnailRenderable(&m_ThumbnailExtents, &m_Geometries),
	m_ThumbnailExtents(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
{
  qDebug("%s",c_VolumeRoverVersionTag);

  //m_UploadBuffer = new unsigned char[256*256*256*4];
  m_UploadBuffer.reset(new unsigned char[256*256*256*4]);

	//m_Geometries.setTransformMode(true);

	m_PendingSocket = 0;

	m_SavedDensityVar = 0;
	m_SavedTimeStep = 0;
	m_SavedZoomedInHandler = 0;
	m_SavedZoomedOutHandler = 0;
	m_Animation = 0;
	m_AnimationTimerId = 0;
	m_FrameNumber = 0;

	m_RGBARendering = false;
	m_WireFrame = false;
	m_TransformGeometry = false;
	m_SubVolumeIsFiltered = false;
	m_AnimationRecording = false;
	m_AnimationPlaying = false;
	m_SaveAnimationFrame = false;

	m_Terminal = new Terminal(NULL);
#ifdef VOLUMEGRIDROVER
#ifdef DETACHED_VOLUMEGRIDROVER
	m_VolumeGridRover = new VolumeGridRover(NULL);
	QAction *ShowVolumeGridRoverAction = new QAction(this,"ShowVolumeGridRoverAction");
	ShowVolumeGridRoverAction->setToggleAction(TRUE);
	ShowVolumeGridRoverAction->addTo(m_ViewMenu);
	connect(ShowVolumeGridRoverAction,
		SIGNAL(toggled(bool)),
		this,
		SLOT(toggleVolumeGridRoverSlot(bool)));
	ShowVolumeGridRoverAction->setText( tr( "Show VolumeGridRover" ) );
	ShowVolumeGridRoverAction->setMenuText( tr( "Show VolumeGridRover" ) );
#else
	m_VolumeGridRover = new VolumeGridRover(m_ViewTabs);
	m_ViewTabs->addTab(m_VolumeGridRover,"Volume Grid Rover");
#endif
	connect(m_VolumeGridRover,SIGNAL(depthChanged(SliceCanvas*,int)),SLOT(gridRoverDepthChangedSlot(SliceCanvas*,int)));
	connect(m_VolumeGridRover,
		SIGNAL(tilingComplete(const boost::shared_ptr<Geometry>&)),
		SLOT(receiveTilingGeometrySlot(const boost::shared_ptr<Geometry>&)));
	connect(m_VolumeGridRover,
		SIGNAL(showImage(const QImage&)),
		SLOT(showImage(const QImage&)));
	connect(m_VolumeGridRover,
		SIGNAL(volumeGenerated(const QString&)),
		SLOT(openFileSlot(const QString&)));
#endif

	connect(m_Terminal, SIGNAL(showToggle(bool)), ShowTerminalAction, SLOT(setOn(bool)));

	m_CacheDir = getCacheDir();

	if (!m_CacheDir.exists("VolumeCache")) {
		m_CacheDir.mkdir("VolumeCache");
		m_CacheDir.cd("VolumeCache");
	}
	else {
		m_CacheDir.cd("VolumeCache");
	}

	m_UpdateMethod = UMDelayed;
	setCaption(QString("Volume Rover"));
	m_RenderServer = 0;
	checkForConnection();

	m_RecentFiles.enable();
	updateVariableInfo(true);

	//m_Toolbar->setStretchableWidget(m_VariableBox);
	
	/* set up the transfer function toolbar */
	m_ColorToolbar = new Q3ToolBar( QString("Transfer Function"), this, Qt::DockBottom );
	m_ColorToolbar->setEnabled( TRUE );
	m_ColorToolbar->setResizeEnabled( TRUE );
	m_ColorToolbar->setMovingEnabled( TRUE );
	m_ColorToolbar->setHorizontallyStretchable( TRUE );
	m_ColorToolbar->setOpaqueMoving( FALSE );

	/* set up the widget stack that contains the transfer function widgets */
	m_ColorToolbarStack = new Q3WidgetStack( m_ColorToolbar, "m_ColorToolbarStack" );

	/* set up the color table (1D transfer func) */
	m_ColorTable = new CVC::ColorTable( m_ColorToolbarStack, "m_ColorTable" );
	m_ColorTable->setSizePolicy( QSizePolicy( QSizePolicy::Preferred, QSizePolicy::Preferred, 0, 0, m_ColorTable->sizePolicy().hasHeightForWidth() ) );
	//m_ColorTable2D = new ColorTable2D( m_ColorToolbarStack, "m_ColorTable2D" );
	//m_ColorTable2D->setSizePolicy( QSizePolicy( QSizePolicy::Preferred, QSizePolicy::Preferred, 0, 0, m_ColorTable2D->sizePolicy().hasHeightForWidth() ) );

	
	//m_ColorToolbar->setStretchableWidget(m_ColorTable);
	connect( m_ColorTable, SIGNAL( functionExploring() ), this, SLOT( functionChangedSlot() ) );
	connect( m_ColorTable, SIGNAL( everythingChanged() ), this, SLOT( isocontourNodesAllChangedSlot() ) );
	connect( m_ColorTable, SIGNAL( everythingChanged() ), this, SLOT( functionChangedSlot() ) );
	connect( m_ColorTable, SIGNAL( functionChanged() ), this, SLOT( functionChangedSlot() ) );
	connect( m_ColorTable, SIGNAL( acquireContourSpectrum() ), this, SLOT( acquireConSpecSlot() ) );
	connect( m_ColorTable, SIGNAL( acquireContourTree() ), this, SLOT( acquireConTreeSlot() ) );
	connect( m_ColorTable, SIGNAL( isocontourNodeAdded(int,double,double,double,double) ), this, SLOT( isocontourNodeAddedSlot(int,double,double,double,double) ) );
	connect( m_ColorTable, SIGNAL( isocontourNodeChanged(int,double) ), this, SLOT( isocontourNodeChangedSlot(int,double) ) );
	connect( m_ColorTable, SIGNAL( isocontourNodeEditRequest(int) ), this, SLOT( isocontourAskIsovalueSlot(int) ) );
	connect( m_ColorTable, SIGNAL( isocontourNodeDeleted(int) ), this, SLOT( isocontourNodeDeletedSlot(int) ) );
	connect( m_ColorTable, SIGNAL( contourTreeNodeAdded(int,int,double) ), this, SLOT( contourTreeNodeAddedSlot(int,int,double) ) );
	connect( m_ColorTable, SIGNAL( contourTreeNodeDeleted(int) ), this, SLOT( contourTreeNodeDeletedSlot(int) ) );
	connect( m_ColorTable, SIGNAL( contourTreeNodeChanged(int,double) ), this, SLOT( contourTreeNodeChangedSlot(int,double) ) );
	m_TransferFunc = TF1D;

	/* add the widgets to the widget stack and bring the color table to the front by default */
	m_ColorToolbarStack->addWidget(m_ColorTable);
	//m_ColorToolbarStack->addWidget(m_ColorTable2D);
	m_ColorToolbarStack->raiseWidget(m_ColorTable);

	//m_MappedVolumeFile = NULL;
	
	m_RemoteSegThread = NULL;
	m_LocalSegThread = NULL;
#ifdef USING_PE_DETECTION
	m_PEDetectionThread = NULL;
#endif

#ifdef USING_POCKET_TUNNEL
	m_PocketTunnelThread = NULL;
#endif

#ifdef USING_MSLEVELSET
	m_MSLevelSetThread = NULL;
#endif
	//m_ProgressBars = new QVBox(NULL);
	//m_ProgressBars->show();

	//m_ThumbnailRenderable.setSliceRendering(true);
#ifdef USING_RECONSTRUCTION
        m_Itercounts = 0;
        Default_newnv = Default_bandwidth = Default_flow = Default_thickness = 0;
        reconManner = 0;
        reconstruction = new Reconstruction();
#endif

}

NewVolumeMainWindow::~NewVolumeMainWindow()
{
	qInstallMsgHandler(0);
	if (m_RenderServer) {
		m_RenderServer->shutdown();
		delete m_RenderServer;
		m_RenderServer = 0;
	}
	
	delete (Terminal*) m_Terminal;
#if defined(VOLUMEGRIDROVER) && defined(DETACHED_VOLUMEGRIDROVER)
	delete m_VolumeGridRover;
#endif

	if(m_RemoteSegThread) delete m_RemoteSegThread;
	if(m_LocalSegThread) delete m_LocalSegThread;
#ifdef USING_PE_DETECTION
	if(m_PEDetectionThread) delete m_PEDetectionThread;
#endif
#ifdef USING_POCKET_TUNNEL
	if(m_PocketTunnelThread) delete m_PocketTunnelThread;
#endif

	//delete m_ProgressBars;

#ifdef USING_RECONSTRUCTION
        delete reconstruction;
#endif
}

void NewVolumeMainWindow::init()
{
  m_ZoomedInRenderable.setVolumeRenderer(new VolumeRenderer);
  m_ThumbnailRenderable.setVolumeRenderer(new VolumeRenderer);
  m_ZoomedInRenderable.addToSimpleOpenGLWidget(*m_ZoomedIn,this,SLOT(mouseReleasedMain()));
  m_ThumbnailRenderable.addToSimpleOpenGLWidget(*m_ZoomedOut,this,SLOT(mouseReleasedPreview()));
  m_ThumbnailRenderable.connectRoverSignals(this, SLOT(explorerMoveSlot()), SLOT(explorerReleaseSlot()));

  // get the background color
  QColor color = getSavedColor();
  m_ZoomedIn->setBackgroundColor(color);
  m_ZoomedInRenderable.setDepthCueColor(Qt::red/255.0, Qt::green/255.0,
					Qt::blue/255.0);
  m_ZoomedInRenderable.setWireCubeColor(1.0-Qt::red,1.0-Qt::green, 1.0-Qt::blue);
  m_ZoomedOut->setBackgroundColor(color);
  m_ThumbnailRenderable.setDepthCueColor(Qt::red/255.0, Qt::green/255.0,Qt::blue/255.0);
  m_ThumbnailRenderable.setRover3DWidgetColor(1.0-Qt::red,1.0-Qt::green, 1.0-Qt::blue);
  //m_ThumbnailRenderable.setSliceRendering(true);
}

void convertendian(float* data, int num) {
	unsigned char * cdata = (unsigned char *) data;
	unsigned char temp;
	for (int i=0; i<num; i++) {
		temp = cdata[i*4+0];
		cdata[i*4+0] = cdata[i*4+3];
		cdata[i*4+3] = temp;

		temp = cdata[i*4+1];
		cdata[i*4+1] = cdata[i*4+2];
		cdata[i*4+2] = temp;
	}

}

unsigned char _map(float num) {
	int intnum;

	intnum = (int)(num / 1000.0 * 255.0);
	intnum = ( intnum<255 ? intnum : 255);
	intnum = ( intnum>0 ? intnum : 0);
	return (unsigned char) intnum;

}

void maptobyte(unsigned char* cdata, float* data, int num) {

	for (int i=0; i<num; i++) {
		cdata[i] = _map(data[i]);
	}
}

void NewVolumeMainWindow::openFile(const QString& filename)
{
  qDebug("Opening: %s",filename.ascii());

#ifdef VOLUMEGRIDROVER
	if(filename.endsWith(".pts"))
	  {
	    SurfRecon::Contour contour;
	    std::ifstream infile(filename.ascii());
	    int num;
	    double x,y,z;

	    while(!infile.eof())
	      {
		infile >> num;

		for(int i=0; i<num; i++)
		  {
		    infile >> x >> y >> z;
		    contour.add(SurfRecon::PointPtr(new SurfRecon::Point(x,y,z)), m_VolumeGridRover->getXYDepth());
		  }

		contour.add(SurfRecon::CurvePtr(new SurfRecon::Curve(m_VolumeGridRover->getXYDepth())));
	      }

	    m_VolumeGridRover->addContour(contour);
	    return;
	  }
#endif

	//lets cheat and convert the spider file to rawiv before loading since the regular volrover i/o doesn't read spider yet
	if(filename.endsWith(".vol")||
	   filename.endsWith(".xmp")||
	   filename.endsWith(".spi"))
	  {
	    QDir tmpdir(m_CacheDir.absPath() + "/tmp");
	    if(!tmpdir.exists()) tmpdir.mkdir(tmpdir.absPath());
	    QFileInfo tmpfile(tmpdir,QFileInfo(filename + ".rawiv").fileName());
	    qDebug("Converting volume %s to %s (%s)",filename.ascii(),tmpfile.absFilePath().ascii(),m_CacheDir.absPath().ascii());
	    
	    QString newfilename;
	    try
	      {
		VolMagick::Volume vol;
		VolMagick::readVolumeFile(vol,filename.toStdString());
		newfilename = QDir::convertSeparators(tmpfile.absFilePath());
		VolMagick::createVolumeFile(newfilename.toStdString(),
					    vol.boundingBox(),
					    vol.dimension(),
					    std::vector<VolMagick::VoxelType>(1, vol.voxelType()));
		VolMagick::writeVolumeFile(vol,newfilename.toStdString());
	      }
	    catch(const VolMagick::Exception& e)
	      {
		QMessageBox::critical(this,"Error opening the file",e.what());
		return;
	      }
	    openFile(newfilename);
	    return;
	  }

	m_CurrentVolumeFilename = filename;
	VolumeFileSource* source = new VolumeFileSource(filename, m_CacheDir.absPath());
	if (source) {
		if (!source->open(this)) {
			QMessageBox::critical( this, "Error opening the file", 
				"An error occured while attempting to open the file: \n"+source->errorReason() );
			delete source;
		}
		else {
#ifdef VOLUMEGRIDROVER
		  printf("filename: %s\n",filename.ascii());
		  
		  try
		    {
		      /* open the file */
		      m_VolumeFileInfo.read(filename);
		      
		      /* set the min and max since they've already been calculated */
		      for(unsigned int i=0; i<source->getNumVars(); i++)
			for(unsigned int j=0; j<source->getNumTimeSteps(); j++)
			  {
			    m_VolumeFileInfo.min(source->getFunctionMinimum(i,j),i,j);
			    m_VolumeFileInfo.max(source->getFunctionMaximum(i,j),i,j);
			  }
		      
		      cout << "Num Variables: " << m_VolumeFileInfo.numVariables() << endl;
		      cout << "Num Timesteps: " << m_VolumeFileInfo.numTimesteps() << endl;
		      cout << "Dimension: " << m_VolumeFileInfo.XDim() << "x" << m_VolumeFileInfo.YDim() << "x" << m_VolumeFileInfo.ZDim() << endl;
		      cout << "Bounding box: ";
		      cout << "(" << m_VolumeFileInfo.boundingBox().minx << "," << m_VolumeFileInfo.boundingBox().miny << "," << m_VolumeFileInfo.boundingBox().minz << ") ";
		      cout << "(" << m_VolumeFileInfo.boundingBox().maxx << "," << m_VolumeFileInfo.boundingBox().maxy << "," << m_VolumeFileInfo.boundingBox().maxz << ") ";
		      cout << endl;
		      cout << "Min voxel value: " << m_VolumeFileInfo.min() << endl;
		      cout << "Max voxel value: " << m_VolumeFileInfo.max() << endl;
		      cout << "Voxel type: " << m_VolumeFileInfo.voxelTypeStr() << endl;
		      cout << "Volume name: " << m_VolumeFileInfo.name() << endl;
		      
		      m_VolumeGridRover->setVolume(m_VolumeFileInfo);
		      m_ThumbnailRenderable.getSliceRenderable()->setVolume(m_VolumeFileInfo);
		      //make sure the slice renderable has the same depths as the volume grid rover
		      m_ThumbnailRenderable.getSliceRenderable()->setDepth(SliceRenderable::XY,
									   m_VolumeGridRover->getXYDepth());
		      m_ThumbnailRenderable.getSliceRenderable()->setDepth(SliceRenderable::XZ,
									   m_VolumeGridRover->getXZDepth());
		      m_ThumbnailRenderable.getSliceRenderable()->setDepth(SliceRenderable::ZY,
									   m_VolumeGridRover->getZYDepth());
		      m_ThumbnailRenderable.getSliceRenderable()->set2DContours(m_VolumeGridRover->getContours());
		      
		      m_ZoomedInRenderable.getSliceRenderable()->setVolume(m_VolumeFileInfo);
		      //make sure the slice renderable has the same depths as the volume grid rover
		      m_ZoomedInRenderable.getSliceRenderable()->setDepth(SliceRenderable::XY,
									   m_VolumeGridRover->getXYDepth());
		      m_ZoomedInRenderable.getSliceRenderable()->setDepth(SliceRenderable::XZ,
									   m_VolumeGridRover->getXZDepth());
		      m_ZoomedInRenderable.getSliceRenderable()->setDepth(SliceRenderable::ZY,
									   m_VolumeGridRover->getZYDepth());
		      m_ZoomedInRenderable.getSliceRenderable()->set2DContours(m_VolumeGridRover->getContours());
#ifdef USING_SKELETONIZATION
		      m_ThumbnailRenderable.getSkeletonRenderable()->setSubVolume(m_VolumeFileInfo.boundingBox());
		      m_ZoomedInRenderable.getSkeletonRenderable()->setSubVolume(m_VolumeFileInfo.boundingBox());
#endif
		    }
		  catch(VolMagick::Exception& e)
		    {
		      QMessageBox::critical(this,"Error opening the file",
					    QString("An error occured while building Volume Grid Rover "
						    "interface to volume file: %1").arg(e.what()));
		      m_VolumeGridRover->unsetVolume();
		      m_ThumbnailRenderable.getSliceRenderable()->unsetVolume();
		      m_ZoomedInRenderable.getSliceRenderable()->unsetVolume();
		    }
#endif
		  
			//updateRecentlyUsedList(filename);
			m_SourceManager.setSource(source);
			if(!m_ZoomedInRenderable.getVolumeBufferManager()->setSourceManager(&m_SourceManager))
			{
				QMessageBox::critical(this, "Error opening the file", 
									"An error occured while attempting to open the file: Error setting source manager");
				delete source;
				return;
			}
			if(!m_ThumbnailRenderable.getVolumeBufferManager()->setSourceManager(&m_SourceManager))
			{
				QMessageBox::critical(this, "Error opening the file", 
									"An error occured while attempting to open the file: Error setting source manager");
				delete source;
				return;
			}

			updateVariableInfo(true);
			// set up the explorer
			getThumbnail();
			//m_DownLoadManager.getThumbnail(&m_SourceManager, getVarNum(), getTimeStep());

			checkError();

			explorerChangedSlot();
			functionChangedSlot();
			// these two funcs may not "get" anything...
			// see acquireCon[Spec/Tree]Slot()
			getContourSpectrum();
			getContourTree();
			
			setUpdateMethod(source->interactiveUpdateHint());
			setCaption(filename + " - Volume Rover");
		}
	}
	else
	  QMessageBox::critical( this, "Error opening the file", 
				"An error occured while attempting to open the file:");
}

void NewVolumeMainWindow::optionsSlot()
{
	OptionsDialog dialog(this, 0, true);
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
		m_ZoomedInRenderable.setDepthCueColor(Qt::red/255.0,Qt::green/255.0, Qt::blue/255.0);
		m_ZoomedInRenderable.setWireCubeColor(1.0-Qt::red,1.0-Qt::green, 1.0-Qt::blue);
		m_ZoomedIn->updateGL();
		m_ZoomedOut->setBackgroundColor(dialog.getColor());
		m_ThumbnailRenderable.setDepthCueColor(Qt::red/255.0,
								Qt::green/255.0,Qt::blue/255.0);
		m_ThumbnailRenderable.setRover3DWidgetColor(1.0-Qt::red,
										1.0-Qt::green, 1.0-Qt::blue);
		m_ZoomedOut->updateGL();
	}
}

void NewVolumeMainWindow::actionSlot()
{
	QString filter = VolumeFileSource::getFilter();
	QString filename = Q3FileDialog::getOpenFileName(QString::null, filter, this);
	//qDebug("actionSlot(): filter == %s",filter.ascii());

	if (!filename.isNull()) {
	/*VolumeFileSource* source = new VolumeFileSource(filename, m_CacheDir.absPath());
	if (source) {
		if (!source->open(this)) {
			QMessageBox::critical( this, "Error opening the file", 
				"An error occured while attempting to open the file: \n"+source->errorReason() );
			delete source;
		}
		else {
			updateRecentlyUsedList(filename);
			m_SourceManager.setSource(source);

			// set up the explorer
			getThumbnail();
			//m_DownLoadManager.getThumbnail(&m_SourceManager, getVarNum(), getTimeStep());

			checkError();

			explorerChangedSlot();
			functionChangedSlot();
			setUpdateMethod(source->interactiveUpdateHint());
			setCaption(filename + " - Volume Rover");
			updateVariableInfo();

		}
	}*/
		openFile(filename);

	}
}

void NewVolumeMainWindow::connectToDCSlot()
{
	if (m_PendingSocket) {
		delete m_PendingSocket;
		m_PendingSocket = NULL;
	}


	m_SourceManager.resetSource();
	
	m_PendingSocket = new Q3Socket();

	connect(m_PendingSocket, SIGNAL(connected()), this, SLOT(finishConnectingToDCSlot()));
	connect(m_PendingSocket, SIGNAL(error(int)), this, SLOT(errorConnectingToDCSlot(int)));

	bool ok = FALSE;
	QString text = QInputDialog::getText(
		tr( "Application name" ),
		tr( "Please enter the server address" ),
		QLineEdit::Normal, tr( "osumed.epn.osc.edu" ), &ok, this );
	if ( ok && !text.isEmpty() )
		m_PendingSocket->connectToHost(text, 7777);
}

void NewVolumeMainWindow::finishConnectingToDCSlot()
{
	DataCutterSource* source = new DataCutterSource(this);

	if (source) {
		QStringList list = source->setSockfd(m_PendingSocket);

		// present the user with the list of files
		FileListDialog listDialog(this, 0, TRUE);

		listDialog.FileList->insertStringList(list);

		if (listDialog.exec() == QDialog::Accepted) {
			QString file = listDialog.FileList->currentText();

			source->setDSName(file);


			disconnect(m_PendingSocket, SIGNAL(connected()), 0, 0);
			disconnect(m_PendingSocket, SIGNAL(error(int)), 0, 0);
			m_PendingSocket = 0;

			m_SourceManager.setSource(source);
			m_ZoomedInRenderable.getVolumeBufferManager()->setSourceManager(&m_SourceManager);
			m_ThumbnailRenderable.getVolumeBufferManager()->setSourceManager(&m_SourceManager);

			setCaption(QString("Data Cutter - Volume Rover"));
			updateVariableInfo(true);

			// set up the explorer
			getThumbnail();
			//m_DownLoadManager.getThumbnail(&m_SourceManager, getVarNum(), getTimeStep());
			
			if (!checkError()) {

				explorerChangedSlot();
				functionChangedSlot();
				setUpdateMethod(source->interactiveUpdateHint());

			}
		}
		else {
			disconnect(m_PendingSocket, SIGNAL(connected()), 0, 0);
			disconnect(m_PendingSocket, SIGNAL(error(int)), 0, 0);
			m_PendingSocket = 0;

			delete source;

		}

	}

}

void NewVolumeMainWindow::errorConnectingToDCSlot(int num)
{
	QString reason;
	switch (num)
	{
	case Q3Socket::ErrConnectionRefused:
		reason = "Connection Refused";
		break;
	case Q3Socket::ErrHostNotFound:
		reason = "Host Not Found";
		break;
	case Q3Socket::ErrSocketRead :
		reason = "Read from socket failed";
		break;
	};
	QMessageBox::critical( this, "Error connecting to the datacutter server", 
		"An error occured while attempting to connect to the datacutter server. \nReason: "+reason );
	delete m_PendingSocket;
	m_PendingSocket = 0;
}

void NewVolumeMainWindow::openFileSlot(const QString& filename)
{
  openFile(filename);
}

void NewVolumeMainWindow::keyPressEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Shift) {
		//qDebug("shift key pressed");

		MouseHandler* handler;
		TrackballRotateInteractor *interactor = new TrackballRotateInteractor;

		// Left Window
		handler = m_ZoomedIn->getMouseHandler(SimpleOpenGLWidget::LeftButtonHandler);
		m_SavedZoomedInHandler = handler->clone();
		handler = m_ZoomedIn->setMouseHandler(SimpleOpenGLWidget::LeftButtonHandler, interactor);
		QObject::connect(handler, SIGNAL(ViewChanged()), this, SLOT(mouseReleasedMain()));
		
		// Right Window
		handler = m_ZoomedOut->getMouseHandler(SimpleOpenGLWidget::LeftButtonHandler);
		m_SavedZoomedOutHandler = handler->clone();
		handler = m_ZoomedOut->setMouseHandler(SimpleOpenGLWidget::LeftButtonHandler, interactor);
		QObject::connect(handler, SIGNAL(ViewChanged()), this, SLOT(mouseReleasedPreview()));

		// don't leak memory
		delete interactor;
	}
#if 0
	else if (e->key() == Qt::Key_M) {

		// Toggle what the mouse handlers manipulate (scene or geometry)
		m_TransformGeometry = !m_TransformGeometry;
		m_ZoomedOut->setObjectManipulationMode(m_TransformGeometry);
		//m_ZoomedIn->setObjectManipulationMode(m_TransformGeometry);

		// swap out the mouse handler for the right button
		if (m_TransformGeometry) {
			MouseHandler* handler;
			ScaleInteractor *interactor = new ScaleInteractor;

			// Right Window
			handler = m_ZoomedOut->getMouseHandler(SimpleOpenGLWidget::MiddleButtonHandler);
			m_SavedZoomedOutHandler = handler->clone();
			handler = m_ZoomedOut->setMouseHandler(SimpleOpenGLWidget::MiddleButtonHandler, interactor);
			QObject::connect(handler, SIGNAL(ViewChanged()), this, SLOT(mouseReleasedMain()));

			delete interactor;
		}
		else { // restore the old mouse handler
			MouseHandler* handler;

			// Right Window
			handler = m_ZoomedOut->getMouseHandler(SimpleOpenGLWidget::MiddleButtonHandler);
			QObject::disconnect(handler, SIGNAL(ViewChanged()), this, SLOT(mouseReleasedMain()));
			m_ZoomedOut->setMouseHandler(SimpleOpenGLWidget::MiddleButtonHandler, m_SavedZoomedOutHandler);

			delete m_SavedZoomedOutHandler;
			m_SavedZoomedOutHandler = NULL;
		}
	}
	else if (e->key() == Qt::Key_Left) {
		if (m_TransformGeometry)
			m_Geometries.incrementActiveObjectIndex();
	}
	else if (e->key() == Qt::Key_Right) {
		if (m_TransformGeometry)
			m_Geometries.decrementActiveObjectIndex();
	}
#endif
}

void NewVolumeMainWindow::keyReleaseEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Shift) {
		//qDebug("shift key released");

		MouseHandler* handler;

		// Left Window
		handler = m_ZoomedIn->getMouseHandler(SimpleOpenGLWidget::LeftButtonHandler);
		QObject::disconnect(handler, SIGNAL(ViewChanged()), this, SLOT(mouseReleasedMain()));
		m_ZoomedIn->setMouseHandler(SimpleOpenGLWidget::LeftButtonHandler, m_SavedZoomedInHandler);
		
		// Right Window
		handler = m_ZoomedOut->getMouseHandler(SimpleOpenGLWidget::LeftButtonHandler);
		QObject::disconnect(handler, SIGNAL(ViewChanged()), this, SLOT(mouseReleasedPreview()));
		m_ZoomedOut->setMouseHandler(SimpleOpenGLWidget::LeftButtonHandler, m_SavedZoomedOutHandler);
		
		// don't leak these
		delete m_SavedZoomedInHandler;
		delete m_SavedZoomedOutHandler;
		m_SavedZoomedInHandler = NULL;
		m_SavedZoomedOutHandler = NULL;
	}
}

void NewVolumeMainWindow::timerEvent(QTimerEvent *e)
{
	// play or record a frame of the animation
	if (e->timerId() == m_AnimationTimerId) {
		//qDebug("got animation timer event");
		if (m_AnimationRecording) {
			// record the frame keys
			recordFrame();
		}
		else if (m_AnimationPlaying) {
			// check for the end of the animation
			if ((m_SaveAnimationFrame
					&& m_FrameNumber*33 > m_Animation->getEndTime())
				|| (!m_SaveAnimationFrame
					&& m_Time.elapsed() > (int) m_Animation->getEndTime()) ) {
				// stop the animation
				stopAnimationSlot();
				return;
			}
			
			unsigned int time = m_Time.elapsed() % m_Animation->getEndTime();
			ViewState state;
			
			if (m_SaveAnimationFrame) {
				time = m_FrameNumber*33;
			}
			
			//m_Animation->getCubicFrame(state, time);
			m_Animation->getFrame(state, time);

			// update the view state for the left window
			m_ZoomedOut->getView().SetOrientation(state.m_Orientation);
			m_ZoomedOut->getView().setTarget(state.m_Target);
			m_ZoomedOut->getView().SetWindowSize(state.m_WindowSize);
			m_ThumbnailRenderable.getVolumeRenderer()->setNearPlane(state.m_ClipPlane);
			m_ExplorerNearPlane->setValue((int)(state.m_ClipPlane*100));
			m_ThumbnailRenderable.getMultiContour()->setWireframeMode(state.m_WireFrame);
			m_ThumbnailRenderable.getGeometryRenderer()->setWireframeMode(state.m_WireFrame);
			m_ZoomedOut->updateGL();

			// update the view state for the right window?
			// write the frame to disk
			if (m_SaveAnimationFrame) {
				QString filename = m_AnimationFrameName 
								+ QString("%1").arg(m_FrameNumber, 5).replace(QChar(' '), QChar('0'))
								+ ".ppm";
				QImage saveImg = m_ZoomedOut->grabFrameBuffer();

				// write the image
				saveImg.save(filename, "PPM");

				// increment the frame number
				m_FrameNumber++;
			}
		}
	}
	else {
		qDebug("caught an orphan timer event");
		killTimer(e->timerId());
	}
}

void NewVolumeMainWindow::functionChangedSlot()
{
	if (!m_RGBARendering) {
		double map[256*4];
		unsigned char byte_map[256*4];
		unsigned int c;
		m_ColorTable->GetTransferFunction(map, 256);
		for (c=0; c<256; c++) {
			byte_map[c*4+0] = mapToChar(map[c*4+0]);
			byte_map[c*4+1] = mapToChar(map[c*4+1]);
			byte_map[c*4+2] = mapToChar(map[c*4+2]);
			//byte_map[c*4+0] = mapToChar(map[c*4+0]*map[c*4+3]);
			//byte_map[c*4+1] = mapToChar(map[c*4+1]*map[c*4+3]);
			//byte_map[c*4+2] = mapToChar(map[c*4+2]*map[c*4+3]);
			byte_map[c*4+3] = mapToChar(map[c*4+3]);
		}
		m_ZoomedIn->makeCurrent();
		m_ZoomedInRenderable.getVolumeRenderer()->uploadColorMap(byte_map);
		m_ZoomedOut->makeCurrent();
		m_ThumbnailRenderable.getVolumeRenderer()->uploadColorMap(byte_map);
		m_ZoomedIn->updateGL();
		m_ZoomedOut->updateGL();
#ifdef VOLUMEGRIDROVER
		m_VolumeGridRover->setTransferFunction(byte_map);
		m_ThumbnailRenderable.getSliceRenderable()->setColorTable(byte_map);
		m_ZoomedInRenderable.getSliceRenderable()->setColorTable(byte_map);
#endif
	}/*

	
		// save out a copy of the colormap
		FILE* fp;
		fp = fopen( "colormap.map", "wb" );
		//fprintf(fp, "unsigned char gColorMap[1024] = {\n");
		//for (int i=0; i<256; i++) {
		//	fprintf(fp, "%d, %d, %d, %d", byte_map[i*4+0],byte_map[i*4+1],
		//									byte_map[i*4+2],byte_map[i*4+3]);
		//	if (i != 255)
		//		fprintf(fp, ",\n");
		//	else
		//		fprintf(fp, "};\n\n");
		//}
			
		fwrite(byte_map, sizeof(unsigned char), 256*4, fp);
		fclose(fp);
	}*/

}

void NewVolumeMainWindow::setExplorerQualitySlot(int value)
{
	m_ThumbnailRenderable.getVolumeRenderer()->setQuality(((double)value)/99.0);
	m_ZoomedOut->updateGL();
}

void NewVolumeMainWindow::setMainQualitySlot(int value)
{
	m_ZoomedInRenderable.getVolumeRenderer()->setQuality(((double)value)/99.0);
	m_ZoomedIn->updateGL();
}

void NewVolumeMainWindow::zoomedInClipSlot(int value)
{
	int clamped = (value>0 ? value : 0);
	clamped = (clamped<99 ? clamped: 99);
	m_ZoomedInRenderable.getVolumeRenderer()->setNearPlane((double)clamped/99.0);
	m_ZoomedIn->updateGL();
}

void NewVolumeMainWindow::zoomedOutClipSlot(int value)
{
	int clamped = (value>0 ? value : 0);
	clamped = (clamped<99 ? clamped: 99);
	m_ThumbnailRenderable.getVolumeRenderer()->setNearPlane((double)clamped/99.0);
	m_ZoomedOut->updateGL();
	
	// set and animation key
	if (m_AnimationRecording)
		recordFrame();
}

void NewVolumeMainWindow::mouseReleasedMain()
{
	if (!m_TransformGeometry)
		m_ZoomedOut->getView().SetOrientation(m_ZoomedIn->getView());
	m_ZoomedOut->updateGL();
	//m_Explorer->setView(CurveView2);
}

void NewVolumeMainWindow::mouseReleasedPreview()
{
	if (!m_TransformGeometry)
		m_ZoomedIn->getView().SetOrientation(m_ZoomedOut->getView());
	m_ZoomedIn->updateGL();
	//CurveView2->setView(m_Explorer);
	
	// set an animation key
	if (m_AnimationRecording)
		recordFrame();
}

void NewVolumeMainWindow::centerSlot()
{
	m_ZoomedOut->getView().setTarget(m_ThumbnailRenderable.getPreviewOrigin());
	m_ZoomedOut->updateGL();

	// set an animation key
	if (m_AnimationRecording)
		recordFrame();
}

inline double transform(double value, double fromMin, double fromMax, double toMin, double toMax) 
{
	return (value-fromMin)*(toMax-toMin)/(fromMax-fromMin) + toMin;
}

void NewVolumeMainWindow::getThumbnail()
{
	if (m_SourceManager.hasSource()) {
		double minX, minY, minZ;
		double maxX, maxY, maxZ;
		minX = m_SourceManager.getMinX();
		minY = m_SourceManager.getMinY();
		minZ = m_SourceManager.getMinZ();

		maxX = m_SourceManager.getMaxX();
		maxY = m_SourceManager.getMaxY();
		maxZ = m_SourceManager.getMaxZ();

		m_ThumbnailExtents.setExtents(minX, maxX, minY, maxY, minZ, maxZ);

		m_ThumbnailRenderable.getVolumeBufferManager()->setRequestRegion(
			m_ThumbnailExtents.getXMin(), m_ThumbnailExtents.getYMin(), m_ThumbnailExtents.getZMin(),
			m_ThumbnailExtents.getXMax(), m_ThumbnailExtents.getYMax(), m_ThumbnailExtents.getZMax(),
			getTimeStep());
		m_ZoomedOut->makeCurrent();
		updateRoverRenderable(&m_ThumbnailRenderable, &m_ThumbnailExtents);
		printf("NewVolumeMainWindow::getThumbnail()\n");
		m_ZoomedOut->updateGL();
	}

}

void NewVolumeMainWindow::getContourSpectrum()
{
	// change the contour spectrum fuction in the transfer function widget
	QString fileName;
	QFile specFile;
	float *isoval,*area,*min_vol,*max_vol,*gradient;

	if ( m_SourceManager.hasSource() ) {
		// get the filename for the spectrum data
		fileName = m_SourceManager.getSource()->getContourSpectrumFileName(getVarNum(), getTimeStep());
		// don't try to open if the filename is null (or non-existant)
		if ( !fileName.isNull() && specFile.exists(fileName) ) {
			// open the file
			specFile.setName(fileName);
			specFile.open(QIODevice::ReadOnly);

			// don't try to read an empty file
			if (specFile.size() == 0) {
				// send NULL data
				m_ColorTable->setSpectrumFunctions(NULL,NULL,NULL,NULL,NULL);
				// we're done
				return;
			}

			specFile.at(0);

			// read the spectrum functions
			isoval = (float *)malloc(256*sizeof(float));
			area = (float *)malloc(256*sizeof(float));
			min_vol = (float *)malloc(256*sizeof(float));
			max_vol = (float *)malloc(256*sizeof(float));
			gradient = (float *)malloc(256*sizeof(float));

			specFile.readBlock((char *)isoval, 256*sizeof(float));
			specFile.readBlock((char *)area, 256*sizeof(float));
			specFile.readBlock((char *)min_vol, 256*sizeof(float));
			specFile.readBlock((char *)max_vol, 256*sizeof(float));
			specFile.readBlock((char *)gradient, 256*sizeof(float));

			// hand the functions over to the ColorTable instance
			m_ColorTable->setSpectrumFunctions(isoval,area,min_vol,max_vol,gradient);
		}
		else if ( !specFile.exists(fileName) ) {
			qDebug("Spectrum file \'%s\' not found!", fileName.latin1());

			// send NULL data
			m_ColorTable->setSpectrumFunctions(NULL,NULL,NULL,NULL,NULL);
		}
	}
}

void NewVolumeMainWindow::getContourTree()
{
	// change the contour tree fuction in the transfer function widget
	QString fileName;
	QFile treeFile;
	int numVerts, numEdges;
	CTVTX *verts;
	CTEDGE *edges;

	if ( m_SourceManager.hasSource() ) {
		// get the filename for the tree data
		fileName = m_SourceManager.getSource()->getContourTreeFileName(getVarNum(), getTimeStep());
		// don't try to open if the filename is null (or non-existant)
		if ( !fileName.isNull() && treeFile.exists(fileName) ) {
			// open the file
			treeFile.setName(fileName);
			treeFile.open(QIODevice::ReadOnly);

			// don't read an empty file... in fact, delete it
			if (treeFile.size() == 0) {
				// delete
				//treeFile.remove();
				// send NULL data
				m_ColorTable->setContourTree(0,0, NULL, NULL);
				// we're done
				return;
			}

			// seek to the beginning
			treeFile.at(0);

			// read the data sizes
			treeFile.readBlock((char *)&numVerts, sizeof(int));
			treeFile.readBlock((char *)&numEdges, sizeof(int));
			// allocate space for the tree
			verts = (CTVTX *)malloc(numVerts*sizeof(CTVTX));
			edges = (CTEDGE *)malloc(numEdges*sizeof(CTEDGE));
			// read the verts and edges
			treeFile.readBlock((char *)verts, numVerts*sizeof(CTVTX));
			treeFile.readBlock((char *)edges, numEdges*sizeof(CTEDGE));

			// hand the tree over to the ColorTable instance
			m_ColorTable->setContourTree(numVerts,numEdges,verts,edges);
		}
		else if ( !treeFile.exists(fileName) ) {
			//qDebug("Contour Tree file \'%s\' not found!", fileName.latin1());
			
			// send NULL data
			m_ColorTable->setContourTree(0,0, NULL, NULL);
		}
	}
}

void NewVolumeMainWindow::acquireConSpecSlot()
{
	qDebug("NewVolumeMainWindow::acquireConSpecSlot()");
	if (m_SourceManager.hasSource()) {
		// compute the contour spectrum
		m_SourceManager.getSource()->computeContourSpectrum((QObject *)this,getVarNum(),getTimeStep());
		// send it to the ColorTable widget
		getContourSpectrum();
	}
	else
		qDebug("NewVolumeMainWindow::acquireConSpecSlot(): no source!");
}

void NewVolumeMainWindow::acquireConTreeSlot()
{
	//qDebug("NewVolumeMainWindow::acquireConSpecSlot()");
	if (m_SourceManager.hasSource()) {
		// compute the contour tree
		m_SourceManager.getSource()->computeContourTree((QObject *)this,getVarNum(),getTimeStep());
		// send it to the ColorTable widget
		getContourTree();
	}
}

void NewVolumeMainWindow::customEvent(QCustomEvent *e)
{
  const char **opStrings = VolMagick::VoxelOperationStatusMessenger::opStrings;
  //static QMap<const VolMagick::Voxels*, QProgressBar*> volMagickProgressBars;
	
  if(e->type() == QEvent::User+100) /* SegmentationFailedEvent */
    QMessageBox::critical(this,"Error",static_cast<SegmentationFailedEvent*>(e)->message(),QMessageBox::Ok,Qt::NoButton,Qt::NoButton);
  else if(e->type() == QEvent::User+101) /* SegmentationFinishedEvent */
    QMessageBox::information(this,"Notice",static_cast<SegmentationFinishedEvent*>(e)->message(),QMessageBox::Ok);

#ifdef USING_PE_DETECTION
  else if(e->type() == QEvent::User+102) /* PEDetectionFailedEvent */
    QMessageBox::critical(this,"Error",static_cast<PEDetectionFailedEvent*>(e)->message());
  else if(e->type() == QEvent::User+103) /* PEDetectionFinishedEvent */
    QMessageBox::information(this,"Notice",static_cast<PEDetectionFinishedEvent*>(e)->message(),QMessageBox::Ok);
#endif

#ifdef USING_POCKET_TUNNEL
  else if(e->type() == QEvent::User+104) /* PocketTunnelFailedEvent */
    QMessageBox::critical(this,"Error",static_cast<PocketTunnelFailedEvent*>(e)->message());
  else if(e->type() == QEvent::User+105) /* PocketTunnelFinishedEvent */
    {
      //m_Geometries.clear();
      m_Geometries.add(new GeometryRenderable(static_cast<PocketTunnelFinishedEvent*>(e)->resultMesh));
      
      m_ZoomedIn->updateGL();
      m_ZoomedOut->updateGL();

      QMessageBox::information(this,"Notice",static_cast<PocketTunnelFinishedEvent*>(e)->message(),QMessageBox::Ok);
    }
#endif

  else if(e->type() == QEvent::User+106) /* TilingInfoEvent */
    {
      m_Geometries.add(new GeometryRenderable(new Geometry(*(static_cast<TilingInfoEvent*>(e)->resultMesh))));

      m_ZoomedIn->updateGL();
      m_ZoomedOut->updateGL();

      //QMessageBox::information(this,"Notice",static_cast<TilingInfoEvent*>(e)->message(),QMessageBox::Ok);
      qDebug("Notice: %s",static_cast<TilingInfoEvent*>(e)->message().ascii());
    }

#ifdef USING_MSLEVELSET
  else if(e->type() == QEvent::User+107) /* MSLevelSetFailedEvent */
    QMessageBox::critical(this,"Error",static_cast<MSLevelSetFailedEvent*>(e)->message());
  else if(e->type() == QEvent::User+108) /* MSLevelSetFinishedEvent */
    {
      static GeometryRenderable *mslevelset_contour = NULL;
      MSLevelSetFinishedEvent *mslsfe = static_cast<MSLevelSetFinishedEvent*>(e);

      //get a pointer to the subvol buffer
      VolumeBuffer *densityBuffer;
      unsigned int xdim, ydim, zdim, len;
      unsigned char *volume;
      
      densityBuffer = m_ZoomedInRenderable.getVolumeBufferManager()->getVolumeBuffer(getVarNum());
      volume = (unsigned char *)densityBuffer->getBuffer();
      xdim = densityBuffer->getWidth();
      ydim = densityBuffer->getHeight();
      zdim = densityBuffer->getDepth();
      len = xdim*ydim*zdim;

      if(xdim != mslsfe->resultVol.XDim() ||
	 ydim != mslsfe->resultVol.YDim() ||
	 zdim != mslsfe->resultVol.ZDim())
	{
	  qDebug("MSLevelSetFinishedEvent: subvolume box dimension changed, not updating it anymore...");
	  return;
	}

      //copy our result to the subvol buffer
      VolMagick::Volume vol(mslsfe->resultVol);
      vol.voxelType(VolMagick::UChar);
      memcpy(volume,*vol,len*sizeof(unsigned char));

      //look in the list of geometries for our GeometryRenderable pointer.  It's possible it
      //could have been deleted, or that we haven't yet created one!
      bool found = false;
      for(unsigned int i = 0; i < m_Geometries.getNumberOfRenderables(); i++)
	if(mslevelset_contour == m_Geometries.get(i))
	  {
	    found = true;
	    break;
	  }
      if(!found)
	{
	  mslevelset_contour = new GeometryRenderable(new Geometry(*mslsfe->resultMesh));
	  m_Geometries.add(mslevelset_contour); //remember, m_Geometry owns its Renderables!
	}

      //copy our result contour to the GeometryRenderable's internal Geometry object.
      *mslevelset_contour->getGeometry() = *mslsfe->resultMesh;

      //refresh the zoomed in view
      m_ZoomedIn->makeCurrent();
      updateRoverRenderable(&m_ZoomedInRenderable, &m_ZoomedInExtents);
	printf("customEvent\n");

      m_ZoomedIn->updateGL();
      
      //QMessageBox::information(this,"Notice",static_cast<MSLevelSetFinishedEvent*>(e)->message(),QMessageBox::Ok);
      qDebug("MSLevelSetFinishedEvent: %s",static_cast<MSLevelSetFinishedEvent*>(e)->message().ascii());
    }  
#endif

  //#if 0
  else if(e->type() == VolMagickOpStartEvent::id)
    {
      VolMagickOpStartEvent *vmose = static_cast<VolMagickOpStartEvent*>(e);
      fprintf(stderr,"Starting %s:\n", opStrings[vmose->operation]);

      //volMagickProgressBars[vmose->voxels] = new QProgressBar(vmose->steps,m_ProgressBars);
      //volMagickProgressBars[vmose->voxels]->setPercentageVisible(true);
      //statusBar()->addWidget(volMagickProgressBars[vmose->voxels]);
    }

  else if(e->type() == VolMagickOpStepEvent::id)
    {
      VolMagickOpStepEvent *vmose = static_cast<VolMagickOpStepEvent*>(e);
      //fprintf(stderr,"%s: %5.2f %%\r",opStrings[vmose->operation],
      //(((float)vmose->currentStep)/((float)((int)(volMagickProgressBars[vmose->voxels]->totalSteps()-1))))*100.0);
      fprintf(stderr,"step...");
      //if(volMagickProgressBars[vmose->voxels])
      //volMagickProgressBars[vmose->voxels]->setProgress(volMagickProgressBars[vmose->voxels]->progress()+1);
    }

  else if(e->type() == VolMagickOpEndEvent::id)
    {
      VolMagickOpEndEvent *vmose = static_cast<VolMagickOpEndEvent*>(e);
      fprintf(stderr,"Finished %s\n", opStrings[vmose->operation]);

      //if(volMagickProgressBars[vmose->voxels])
      //{
	  //statusBar()->removeWidget(volMagickProgressBars[vmose->voxels]);
      //delete volMagickProgressBars[vmose->voxels];
      //volMagickProgressBars.remove(vmose->voxels);
      //}
    }
  else if(dynamic_cast<SignedDistanceFunctionDialogGrabSubVolBoxEvent*>(e))
    {
      SignedDistanceFunctionDialogGrabSubVolBoxEvent *sdfgs = 
	static_cast<SignedDistanceFunctionDialogGrabSubVolBoxEvent*>(e);
      SignedDistanceFunctionDialog *sdfd = sdfgs->dialog;
      sdfd->setBoundingBox(m_ZoomedInExtents.getXMin(), 
			   m_ZoomedInExtents.getYMin(), 
			   m_ZoomedInExtents.getZMin(),
			   m_ZoomedInExtents.getXMax(),
			   m_ZoomedInExtents.getYMax(),
			   m_ZoomedInExtents.getZMax());
    }
  //#endif
}

void NewVolumeMainWindow::explorerChangedSlot()
{
	if (m_SourceManager.hasSource()) {
		Extents subVolume = m_ThumbnailRenderable.getSubVolume();
		Extents boundary = m_ThumbnailRenderable.getBoundary();

		double minX, minY, minZ;
		double maxX, maxY, maxZ;
		minX = transform(subVolume.getXMin(), 
			boundary.getXMin(), boundary.getXMax(),
			m_SourceManager.getMinX(), m_SourceManager.getMaxX());
		minY = transform(subVolume.getYMin(), 
			boundary.getYMin(), boundary.getYMax(),
			m_SourceManager.getMinY(), m_SourceManager.getMaxY());
		minZ = transform(subVolume.getZMin(), 
			boundary.getZMin(), boundary.getZMax(),
			m_SourceManager.getMinZ(), m_SourceManager.getMaxZ());

		maxX = transform(subVolume.getXMax(), 
			boundary.getXMin(), boundary.getXMax(),
			m_SourceManager.getMinX(), m_SourceManager.getMaxX());
		maxY = transform(subVolume.getYMax(), 
			boundary.getYMin(), boundary.getYMax(),
			m_SourceManager.getMinY(), m_SourceManager.getMaxY());
		maxZ = transform(subVolume.getZMax(), 
			boundary.getZMin(), boundary.getZMax(),
			m_SourceManager.getMinZ(), m_SourceManager.getMaxZ());

		m_ZoomedInExtents.setExtents(minX, maxX, minY, maxY, minZ, maxZ);
		m_ZoomedInRenderable.getVolumeBufferManager()->setRequestRegion(
			m_ZoomedInExtents.getXMin(), m_ZoomedInExtents.getYMin(), m_ZoomedInExtents.getZMin(),
			m_ZoomedInExtents.getXMax(), m_ZoomedInExtents.getYMax(), m_ZoomedInExtents.getZMax(),
			getTimeStep());

#ifdef VOLUMEGRIDROVER
		m_ZoomedInRenderable.getSliceRenderable()->setSubVolume(VolMagick::BoundingBox(minX,minY,minZ,
											       maxX,maxY,maxZ));
#endif 
#ifdef USING_SKELETONIZATION
		m_ZoomedInRenderable.getSkeletonRenderable()->setSubVolume(VolMagick::BoundingBox(minX,minY,minZ,
												  maxX,maxY,maxZ));
#endif
		m_ZoomedIn->makeCurrent();
		updateRoverRenderable(&m_ZoomedInRenderable, &m_ZoomedInExtents);
		printf("NewVolumeMainWindow::explorerChangedSlot()\n");
		m_ZoomedIn->updateGL();

		checkError();
	}
}

void NewVolumeMainWindow::explorerMoveSlot()
{
	if (m_UpdateMethod == UMInteractive) {
		explorerChangedSlot();
		// filtering operations are erased
		m_SubVolumeIsFiltered = false;
	}
}

void NewVolumeMainWindow::explorerReleaseSlot()
{
	if (m_UpdateMethod == UMDelayed) {
		explorerChangedSlot();
		// filtering operations are erased
		m_SubVolumeIsFiltered = false;
	}
}

void NewVolumeMainWindow::variableOrTimeChangeSlot()
{
	// filtering operations are erased
	m_SubVolumeIsFiltered = false;

	getThumbnail();
	explorerChangedSlot();
	getContourSpectrum();
}

void NewVolumeMainWindow::toggleWireframeRenderingSlot(bool state)
{
	m_ThumbnailRenderable.getMultiContour()->setWireframeMode(state);
	m_ThumbnailRenderable.getGeometryRenderer()->setWireframeMode(state);
	m_ZoomedOut->updateGL();
	m_ZoomedInRenderable.getMultiContour()->setWireframeMode(state);
	m_ZoomedInRenderable.getGeometryRenderer()->setWireframeMode(state);
	m_ZoomedIn->updateGL();

	// set our internal flag (used by the animation code)
	m_WireFrame = state;

	// add an animation keyframe
	if (m_AnimationRecording)
		recordFrame();
}

void NewVolumeMainWindow::toggleRenderSurfaceWithWireframeSlot(bool state)
{
	m_ThumbnailRenderable.getMultiContour()->setSurfWithWire(state);
	m_ThumbnailRenderable.getGeometryRenderer()->setSurfWithWire(state);
	m_ZoomedOut->updateGL();
	m_ZoomedInRenderable.getMultiContour()->setSurfWithWire(state);
	m_ZoomedInRenderable.getGeometryRenderer()->setSurfWithWire(state);
	m_ZoomedIn->updateGL();

	// set our internal flag (used by the animation code)
	m_WireFrame = state;

	// add an animation keyframe
	if (m_AnimationRecording)
		recordFrame();
}

void NewVolumeMainWindow::toggleWireCubeSlot(bool state)
{
	m_ThumbnailRenderable.toggleWireCubeDrawing(state);
	m_ZoomedOut->updateGL();
	m_ZoomedInRenderable.toggleWireCubeDrawing(state);
	m_ZoomedIn->updateGL();
}

void NewVolumeMainWindow::toggleDepthCueSlot(bool state)
{
	m_ThumbnailRenderable.toggleDepthCueing(state);
	m_ZoomedOut->updateGL();
	m_ZoomedInRenderable.toggleDepthCueing(state);
	m_ZoomedIn->updateGL();
}

void NewVolumeMainWindow::isocontourNodeChangedSlot(int node, double value)
{

	qDebug("Change %d, %lf", node, value*255.0);
	//if (m_DownLoadManager.hasSource()) {
		QTime t;
		t.start();
		m_ThumbnailRenderable.getMultiContour()->setIsovalue(node, value*255.0);
		m_ZoomedOut->updateGL();

		qDebug("marching cubes :  %d\n", t.elapsed() );
		QTime t1;
		t1.start();
		m_ZoomedInRenderable.getMultiContour()->setIsovalue(node, value*255.0);
		m_ZoomedIn->updateGL();
		qDebug("marching cubes :  %d\n", t1.elapsed() );
	//}

	syncIsocontourValuesWithVolumeGridRover();
}

void NewVolumeMainWindow::isocontourAskIsovalueSlot(int node)
{
	bool ok;
  	if (m_SourceManager.hasSource()) {
    	double isoval = QInputDialog::getDouble(
            "Iso contour", "Enter an iso-value (actual intensity):", 0.0, -2147483647, 2147483647, 4, &ok, this);
    	if ( ok ) {
		//convert isoval to value
		double valRange = m_VolumeFileInfo.max() - m_VolumeFileInfo.min();
		float value = (isoval - m_VolumeFileInfo.min())/valRange;
		qDebug("Change %d , %lf (actual: %f)", node, value*255.0, isoval);

		m_ColorTable->moveIsocontourNode(node,value);
    		}
	}
}

void NewVolumeMainWindow::isocontourNodeColorChangedSlot(int node, double R, double G, double B)
{
	qDebug("Change %d, (%lf,%lf,%lf)", node, R, G, B);
	//if (m_DownLoadManager.hasSource()) {
		m_ThumbnailRenderable.getMultiContour()->setColor(node, R,G,B);
		m_ZoomedOut->updateGL();

		m_ZoomedInRenderable.getMultiContour()->setColor(node, R,G,B);
		m_ZoomedIn->updateGL();
	//}

	syncIsocontourValuesWithVolumeGridRover();}

void NewVolumeMainWindow::isocontourNodeAddedSlot(int node, double value, double R, double G, double B)
{
	
	qDebug("Change %d, %lf", node, value*255.0);
	//if (m_DownLoadManager.hasSource()) {
		QTime t;
		t.start();

		m_ThumbnailRenderable.getMultiContour()->addContour(node, value*255.0, R,G,B);
		m_ZoomedOut->updateGL();

		qDebug("marching cubes :  %d\n", t.elapsed() );
		QTime t1;
		t1.start();
		m_ZoomedInRenderable.getMultiContour()->addContour(node, value*255.0, R,G,B);
		m_ZoomedIn->updateGL();
		qDebug("marching cubes :  %d\n", t1.elapsed() );
	//}
	
	syncIsocontourValuesWithVolumeGridRover();
}

void NewVolumeMainWindow::isocontourNodeDeletedSlot(int node)
{
	qDebug("Deleted %d", node);


	m_ThumbnailRenderable.getMultiContour()->removeContour(node);
	m_ZoomedOut->updateGL();

	
	m_ZoomedInRenderable.getMultiContour()->removeContour(node);
	m_ZoomedIn->updateGL();

	syncIsocontourValuesWithVolumeGridRover();
}

void NewVolumeMainWindow::isocontourNodesAllChangedSlot()
{
	qDebug("All contours changed");

	int c, id;
	double value, red, blue, green;

	m_ThumbnailRenderable.getMultiContour()->removeAll();
	m_ZoomedInRenderable.getMultiContour()->removeAll();

	for (c=0; c<m_ColorTable->getIsocontourMap().GetSize(); c++) {
		id = m_ColorTable->getIsocontourMap().GetIDofIthNode(c);
		value = m_ColorTable->getIsocontourMap().GetPositionOfIthNode(c);
		red = m_ColorTable->getIsocontourMap().GetRed(c);
		green = m_ColorTable->getIsocontourMap().GetGreen(c);
		blue = m_ColorTable->getIsocontourMap().GetBlue(c);
#if 0
		m_ThumbnailRenderable.getMultiContour()->addContour(id, value*255.0, 0.25,0.25,0.25);
		m_ZoomedInRenderable.getMultiContour()->addContour(id, value*255.0, 0.25,0.25,0.25);
#endif
		m_ThumbnailRenderable.getMultiContour()->addContour(id, value*255.0, red, green, blue);
		m_ZoomedInRenderable.getMultiContour()->addContour(id, value*255.0, red, green, blue);
	}

	syncIsocontourValuesWithVolumeGridRover();
	
	m_ZoomedOut->updateGL();
	m_ZoomedIn->updateGL();

}

void NewVolumeMainWindow::syncIsocontourValuesWithVolumeGridRover()
{
#ifdef VOLUMEGRIDROVER
  VolumeGridRover::IsocontourValues vals;
  for(int c=0; c<m_ColorTable->getIsocontourMap().GetSize(); c++)
    {
      VolumeGridRover::IsocontourValue val;
      val.value = 
	m_VolumeFileInfo.min() + 
	m_ColorTable->getIsocontourMap().GetPositionOfIthNode(c)*(m_VolumeFileInfo.max()-m_VolumeFileInfo.min());
      val.red = m_ColorTable->getIsocontourMap().GetRed(c);
      val.green = m_ColorTable->getIsocontourMap().GetGreen(c);
      val.blue = m_ColorTable->getIsocontourMap().GetBlue(c);
      vals.push_back(val);
    }
  m_VolumeGridRover->isocontourValues(vals);
#endif
}

void NewVolumeMainWindow::contourTreeNodeChangedSlot(int node, double value)
{
	qDebug("contourTreeNodeChangedSlot(%d,%f)", node, value);
}

void NewVolumeMainWindow::contourTreeNodeAddedSlot(int node, int edge, double value)
{
	qDebug("contourTreeNodeAddedSlot(%d,%d,%f)", node, edge, value);
}

void NewVolumeMainWindow::contourTreeNodeDeletedSlot(int node)
{
	qDebug("contourTreeNodeDeletedSlot(%d)", node);
}

void NewVolumeMainWindow::connectServerSlot()
{
#ifdef USINGCORBA
	ServerSelectorDialog dialog(this);

	if (dialog.exec() == QDialog::Accepted) {
		disconnectServerSlot();

		QFile reffile;

		if (dialog.m_ServerListBox->currentText() == "Raycasting Server") {
			m_RenderServer = new RaycastRenderServer();
			reffile.setName("volserv.ref");
		}
		else if (dialog.m_ServerListBox->currentText() == "Texturebased Server") {
			m_RenderServer = new TextureRenderServer();	
			reffile.setName("3DTex.ref");
		}
		else {
			m_RenderServer = new IsocontourRenderServer();
			reffile.setName("name.ref");
		}

		reffile.open(QIODevice::ReadOnly);
		Q3TextStream stream(&reffile);
		char buffer[512];
		stream >> (char*)buffer;
		
		if (!m_RenderServer->init(buffer, this)) {
			QMessageBox::critical( this, "Error connecting to server", 
				"An error occured while connecting to the rendering server.");
			// destroy the bad connection
			disconnectServerSlot(); 
		}
	}
	checkForConnection();
#endif
}

void NewVolumeMainWindow::disconnectServerSlot()
{
	if (m_RenderServer) {
		m_RenderServer->shutdown();
		delete m_RenderServer;
		m_RenderServer = 0;
	}
	checkForConnection();
}

void NewVolumeMainWindow::serverSettingsSlot()
{
	if (m_RenderServer) {
		if (!m_RenderServer->serverSettings(this)) {
			QMessageBox::critical( this, "Error changing the settings", 
				"An error occured while changing the settings on the rendering server.");
		}
	}
}

void NewVolumeMainWindow::renderFrameSlot()
{
	if (m_RenderServer) {
		// get the view information and add the clip plane position (ugly)
		ViewInformation info = m_ZoomedOut->getViewInformation();
		info.setClipPlane((float)m_ThumbnailRenderable.getVolumeRenderer()->getNearPlane());
		// call the render server
		QPixmap* pixmap = m_RenderServer->renderImage(
			FrameInformation(info, m_ColorTable->getColorTableInformation())
			);
		if (pixmap) {
			ImageViewer* iv = new ImageViewer();
			iv->setPixmap(*pixmap);
			delete pixmap;
			iv->show();
			connect(this,SIGNAL(destroyed()),iv,SLOT(deleteLater()));
		}
		else {
			QMessageBox::critical( this, "Error rendering", 
				"An error occured while rendering the image on the rendering server.");
		}
	}
}

void NewVolumeMainWindow::renderAnimationSlot()
{
	// ignore for now, we will create this later
}

void NewVolumeMainWindow::bilateralFilterSlot()
{
  if (m_SourceManager.hasSource()) {
    if (!m_RGBARendering) {
      BilateralFilterDialog dialog(this);

      dialog.m_RadSigEdit->setValidator(new QDoubleValidator(this));
      dialog.m_SpatSigEdit->setValidator(new QDoubleValidator(this));
      dialog.m_FilRadEdit->setValidator(new QDoubleValidator(this));

      if(dialog.exec() == QDialog::Accepted) {

	/*
	  Check if we are doing in place filtering of the actual volume data file
	  instead of simply the current subvolume buffer
	  TODO: need to implement out-of-core filtering using VolMagick
	*/
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

}

void NewVolumeMainWindow::gdtvFilterSlot()
{
  if(m_SourceManager.hasSource()){
    
    if(!m_RGBARendering){
      GDTVFilterDialog dialog(this);
          
      printf("begin gdtvfiltering ...\n");
      dialog.m_ParaterqEdit->setValidator(new QDoubleValidator(this));
      dialog.m_LambdaEdit->setValidator(new QDoubleValidator(this));
      dialog.m_IterationEdit->setValidator(new QIntValidator(this));
      dialog.m_NeigbourEdit->setValidator(new QIntValidator(this));
      
      if(dialog.exec()==QDialog::Accepted){
#if 0
	VolMagick::Volume vol;
	VolMagick::readVolumeFile(vol, m_VolumeFileInfo.filename());
	vol = gdtvFilter(vol, dialog.m_ParaterqEdit->text().toDouble(),
			 dialog.m_LambdaEdit->text().toDouble(),
			 dialog.m_IterationEdit->text().toInt(),
			 dialog.m_NeigbourEdit->text().toInt());
	VolMagick::writeVolumeFile(vol,m_VolumeFileInfo.filename());
#endif

	/*
	  Check if we are doing in place filtering of the actual volume data file
	  instead of simply the current subvolume buffer
	  TODO: need to implement out-of-core filtering using VolMagick
	*/
	if(!dialog.m_Preview->isChecked())
	  {
	    if(QMessageBox::warning(this,
				    "GDTV Filter",
				    "Are you sure you want to do this?\n"
				    "This operation will change the current loaded volume file\n"
				    "and cannot be un-done!",
				    QMessageBox::Cancel | QMessageBox::Default | QMessageBox::Escape,
				    QMessageBox::Ok) != QMessageBox::Ok)
	      return;

	    VolMagickOpStatus opstatus;
	    VolMagick::Volume vol;
	    vol.messenger(&opstatus); //so we get a nice progress dialog during filtering, which will take a while
	    VolMagick::readVolumeFile(vol,
				      m_VolumeFileInfo.filename(),
				      getVarNum(),
				      getTimeStep());
	    vol.gdtvFilter(dialog.m_ParaterqEdit->text().toDouble(),
			   dialog.m_LambdaEdit->text().toDouble(),
			   dialog.m_IterationEdit->text().toInt(),
			   dialog.m_NeigbourEdit->text().toInt());
	    VolMagick::writeVolumeFile(vol,
				       m_VolumeFileInfo.filename(),
				       getVarNum(),
				       getTimeStep());
	    QString tmpp(m_VolumeFileInfo.filename().c_str());
	    openFile(tmpp);
	    return;
	  }

	//printf("gdtvfiltering is over ...\n");
	//openFile(m_VolumeFileInfo.filename());
	//return; 	    
	
	VolumeBuffer *densityBuffer;
	unsigned int xdim, ydim, zdim, i, len;
	unsigned char *volume;
	
	densityBuffer = m_ZoomedInRenderable.getVolumeBufferManager()->getVolumeBuffer(getVarNum());
	volume = (unsigned char *)densityBuffer->getBuffer();
	xdim = densityBuffer->getWidth();
	ydim = densityBuffer->getHeight();
	zdim = densityBuffer->getDepth();
	len = xdim*ydim*zdim;
	
	VolMagickOpStatus opstatus;
	VolMagick::Voxels vox(VolMagick::Dimension(xdim,ydim,zdim),VolMagick::UChar);
	vox.messenger(&opstatus); //so we get a nice progress dialog during filtering, which will take a while
	memcpy(*vox,volume,len*sizeof(unsigned char));
	
	try
	  {
	    vox.gdtvFilter(dialog.m_ParaterqEdit->text().toDouble(),
			   dialog.m_LambdaEdit->text().toDouble(),
			   dialog.m_IterationEdit->text().toInt(),
			   dialog.m_NeigbourEdit->text().toInt());
	  }
	catch(VolMagickOpStatus::OperationCancelled &e)
	  {
	    qDebug("GDTV Filtering operation cancelled.");
	    return;
	  }
	
	memcpy(volume,*vox,len*sizeof(unsigned char));
	
	m_ZoomedIn->makeCurrent();
	updateRoverRenderable(&m_ZoomedInRenderable, &m_ZoomedInExtents);
	printf("NewVolumeMainWindow::gdtvFilterSlot()\n");
	m_ZoomedIn->updateGL();
	
	m_SubVolumeIsFiltered = true;
      }
    }
    else
      {
	QMessageBox::warning(this, tr("GDTV Filter"),
			     tr("This feature is not available for RGBA volumes."), 0,0);
      }
  }
  else
    {
      QMessageBox::warning(this, tr("GDTV Filter"),
			   tr("A volume must be loaded for this feature to work."), 0,0);
    }
}

void NewVolumeMainWindow::colorGeometryByVolumeSlot()
{
  if(m_Geometries.getNumberOfRenderables() == 0)
    {
      QMessageBox::information(this, "Color Geometry By Volume Slot", "Load a geometry file first.");
      return;
    }

  //if rgba rendering, just grab colors from the volume directly
  //note, geometry doesn't support alpha :(
  if(m_RGBARendering)
    {
      VolumeBuffer* redBuffer = m_ZoomedInRenderable.getVolumeBufferManager()->getVolumeBuffer(getRedVariable());
      VolumeBuffer* greenBuffer = m_ZoomedInRenderable.getVolumeBufferManager()->getVolumeBuffer(getGreenVariable());
      VolumeBuffer* blueBuffer = m_ZoomedInRenderable.getVolumeBufferManager()->getVolumeBuffer(getBlueVariable());
      //VolumeBuffer* alphaBuffer = m_ZoomedInRenderable->getVolumeBufferManager()->getVolumeBuffer(getAlphaVariable());
      unsigned int xdim, ydim, zdim, len;
      unsigned char *volume;

      xdim = redBuffer->getWidth();
      ydim = redBuffer->getHeight();
      zdim = redBuffer->getDepth();
      len = xdim*ydim*zdim;
      volume = (unsigned char *)redBuffer->getBuffer();
      VolMagick::Volume red_vol(VolMagick::Dimension(xdim,ydim,zdim),
				VolMagick::UChar,
				VolMagick::BoundingBox(m_ZoomedInExtents.getXMin(), 
						       m_ZoomedInExtents.getYMin(), 
						       m_ZoomedInExtents.getZMin(),
						       m_ZoomedInExtents.getXMax(),
						       m_ZoomedInExtents.getYMax(),
						       m_ZoomedInExtents.getZMax()));
      memcpy(*red_vol,volume,len*sizeof(unsigned char));

      xdim = greenBuffer->getWidth();
      ydim = greenBuffer->getHeight();
      zdim = greenBuffer->getDepth();
      len = xdim*ydim*zdim;
      volume = (unsigned char *)greenBuffer->getBuffer();
      VolMagick::Volume green_vol(VolMagick::Dimension(xdim,ydim,zdim),
				  VolMagick::UChar,
				  VolMagick::BoundingBox(m_ZoomedInExtents.getXMin(), 
							 m_ZoomedInExtents.getYMin(), 
							 m_ZoomedInExtents.getZMin(),
							 m_ZoomedInExtents.getXMax(),
							 m_ZoomedInExtents.getYMax(),
							 m_ZoomedInExtents.getZMax()));
      memcpy(*green_vol,volume,len*sizeof(unsigned char));

      xdim = blueBuffer->getWidth();
      ydim = blueBuffer->getHeight();
      zdim = blueBuffer->getDepth();
      len = xdim*ydim*zdim;
      volume = (unsigned char *)blueBuffer->getBuffer();
      VolMagick::Volume blue_vol(VolMagick::Dimension(xdim,ydim,zdim),
				 VolMagick::UChar,
				 VolMagick::BoundingBox(m_ZoomedInExtents.getXMin(), 
							m_ZoomedInExtents.getYMin(), 
							m_ZoomedInExtents.getZMin(),
							m_ZoomedInExtents.getXMax(),
							m_ZoomedInExtents.getYMax(),
							m_ZoomedInExtents.getZMax()));
      memcpy(*blue_vol,volume,len*sizeof(unsigned char));

      for(unsigned int i = 0; i < m_Geometries.getNumberOfRenderables(); i++)
	{
	  GeometryRenderable *gr = static_cast<GeometryRenderable*>(m_Geometries.getIth(i));
	  Geometry *geom = gr->getGeometry();
	  
	  geom->AllocatePointColors();
	  for(unsigned int j = 0; j < geom->m_NumPoints; j++)
	    {
	      //if the geometry falls outside of the box, use the background color!
	      try
		{
		  geom->m_PointColors[j*3+0] = red_vol.interpolate(geom->m_Points[j*3+0],
								   geom->m_Points[j*3+1],
								   geom->m_Points[j*3+2])/255.0;
		  geom->m_PointColors[j*3+1] = green_vol.interpolate(geom->m_Points[j*3+0],
								     geom->m_Points[j*3+1],
								     geom->m_Points[j*3+2])/255.0;
		  geom->m_PointColors[j*3+1] = blue_vol.interpolate(geom->m_Points[j*3+0],
								    geom->m_Points[j*3+1],
								    geom->m_Points[j*3+2])/255.0;
		}
	      catch(VolMagick::Exception &e)
		{
		  geom->m_PointColors[j*3+0] = double(m_ZoomedIn->getBackgroundColor().red())/255.0;
		  geom->m_PointColors[j*3+1] = double(m_ZoomedIn->getBackgroundColor().green())/255.0;
		  geom->m_PointColors[j*3+2] = double(m_ZoomedIn->getBackgroundColor().blue())/255.0;
		}
	    }

	}
    }
  else
    {
      VolumeBuffer *densityBuffer;
      unsigned int xdim, ydim, zdim, len;
      unsigned char *volume;
      densityBuffer = m_ZoomedInRenderable.getVolumeBufferManager()->getVolumeBuffer(getVarNum());
      volume = (unsigned char *)densityBuffer->getBuffer();
      xdim = densityBuffer->getWidth();
      ydim = densityBuffer->getHeight();
      zdim = densityBuffer->getDepth();
      len = xdim*ydim*zdim;
      VolMagick::Volume vol(VolMagick::Dimension(xdim,ydim,zdim),
			    VolMagick::UChar,
			    VolMagick::BoundingBox(m_ZoomedInExtents.getXMin(), 
						   m_ZoomedInExtents.getYMin(), 
						   m_ZoomedInExtents.getZMin(),
						   m_ZoomedInExtents.getXMax(),
						   m_ZoomedInExtents.getYMax(),
						   m_ZoomedInExtents.getZMax()));
      memcpy(*vol,volume,len*sizeof(unsigned char));

      //use the colors from the transfer function
      double map[256*4];
      m_ColorTable->GetTransferFunction(map, 256);

      for(unsigned int i = 0; i < m_Geometries.getNumberOfRenderables(); i++)
	{
	  GeometryRenderable *gr = static_cast<GeometryRenderable*>(m_Geometries.getIth(i));
	  Geometry *geom = gr->getGeometry();
	  
	  geom->AllocatePointColors();
	  for(unsigned int j = 0; j < geom->m_NumPoints; j++)
	    {
	      //if the geometry falls outside of the box, use the background color!
	      try
		{
		  double val = vol.interpolate(geom->m_Points[j*3+0],
					       geom->m_Points[j*3+1],
					       geom->m_Points[j*3+2]);
		  int int_val = int(val);
		  geom->m_PointColors[j*3+0] = map[int_val*4+0];
		  geom->m_PointColors[j*3+1] = map[int_val*4+1];
		  geom->m_PointColors[j*3+2] = map[int_val*4+2];
		}
	      catch(VolMagick::Exception &e)
		{
		  geom->m_PointColors[j*3+0] = double(m_ZoomedIn->getBackgroundColor().red())/255.0;
		  geom->m_PointColors[j*3+1] = double(m_ZoomedIn->getBackgroundColor().green())/255.0;
		  geom->m_PointColors[j*3+2] = double(m_ZoomedIn->getBackgroundColor().blue())/255.0;
		}
	    }
	}
    }

  m_ZoomedIn->updateGL();
  m_ZoomedOut->updateGL();
}

//removes geometry that is outside of the subvolume box.  Outside
//is defined by any verts part of an object being outside of the
//box.  This doesn't yet attempt to break up geometry such that
//objects that are partially outside of the box are cut up to fit.
void NewVolumeMainWindow::cullGeometryWithSubvolumeBoxSlot()
{
  typedef cvcraw_geometry::geometry_t::point_t point_t;
  typedef cvcraw_geometry::geometry_t::color_t color_t;
  typedef cvcraw_geometry::geometry_t::line_t line_t;
  typedef cvcraw_geometry::geometry_t::triangle_t triangle_t;
  typedef cvcraw_geometry::geometry_t::quad_t quad_t;

  VolMagick::BoundingBox zoomed_box(m_ZoomedInExtents.getXMin(), 
				    m_ZoomedInExtents.getYMin(), 
				    m_ZoomedInExtents.getZMin(),
				    m_ZoomedInExtents.getXMax(),
				    m_ZoomedInExtents.getYMax(),
				    m_ZoomedInExtents.getZMax());
  
  for(unsigned int i = 0; i < m_Geometries.getNumberOfRenderables(); i++)
    {
      Geometry *geo = static_cast<GeometryRenderable*>(m_Geometries.getIth(i))->getGeometry();
      cvcraw_geometry::geometry_t geometry;
      if(geo->m_GeoFrame)
	{
	  for(unsigned int j = 0; j < geo->m_GeoFrame->numverts; j++)
	    {
	      point_t vert = {{ geo->m_GeoFrame->verts[j][0],
				geo->m_GeoFrame->verts[j][1],
				geo->m_GeoFrame->verts[j][2] }};
	      geometry.points.push_back(vert);
	      geometry.boundary.push_back(geo->m_GeoFrame->bound_sign[j]);
	    }
	  for(unsigned int j = 0; j < geo->m_GeoFrame->numtris; j++)
	    {
	      triangle_t tri = {{ geo->m_GeoFrame->triangles[j][0],
				  geo->m_GeoFrame->triangles[j][1],
				  geo->m_GeoFrame->triangles[j][2] }};

	      //check if this triangle is inside the box.  if it is,
	      //add it to the set
	      boost::dynamic_bitset<> inside;
	      for(unsigned int k = 0; k < 3; k++)
		if(zoomed_box.minx <= geometry.points[tri[k]][0] &&
		   zoomed_box.miny <= geometry.points[tri[k]][1] &&
		   zoomed_box.minz <= geometry.points[tri[k]][2] &&
		   zoomed_box.maxx >= geometry.points[tri[k]][0] &&
		   zoomed_box.maxx >= geometry.points[tri[k]][1] &&
		   zoomed_box.maxx >= geometry.points[tri[k]][2])
		  inside.push_back(true);
	      if(inside.size()==3)
		geometry.tris.push_back(tri);
	    }
	  for(unsigned int j = 0; j < geo->m_GeoFrame->numquads; j++)
	    {
	      quad_t quad = {{ geo->m_GeoFrame->quads[j][0],
			       geo->m_GeoFrame->quads[j][1],
			       geo->m_GeoFrame->quads[j][2],
			       geo->m_GeoFrame->quads[j][3] }};

	      //check if this quad is inside the box.  if it is,
	      //add it to the set
	      boost::dynamic_bitset<> inside;
	      for(unsigned int k = 0; k < 4; k++)
		if(zoomed_box.minx <= geometry.points[quad[k]][0] &&
		   zoomed_box.miny <= geometry.points[quad[k]][1] &&
		   zoomed_box.minz <= geometry.points[quad[k]][2] &&
		   zoomed_box.maxx >= geometry.points[quad[k]][0] &&
		   zoomed_box.maxx >= geometry.points[quad[k]][1] &&
		   zoomed_box.maxx >= geometry.points[quad[k]][2])
		  inside.push_back(true);
	      if(inside.size()==4)
		geometry.quads.push_back(quad);
	    }
	}
      else
	{
	  for(unsigned int j = 0; j < geo->m_NumPoints; j++)
	    {
	      point_t vert = {{ geo->m_Points[j*3+0],
				geo->m_Points[j*3+1],
				geo->m_Points[j*3+2] }};
	      geometry.points.push_back(vert);
	    }
	  if(geo->m_PointColors)
	    for(unsigned int j = 0; j < geo->m_NumPoints; j++)
	      {
		color_t color = {{ geo->m_PointColors[j*3+0],
				   geo->m_PointColors[j*3+1],
				   geo->m_PointColors[j*3+2] }};
		geometry.colors.push_back(color);
	      }
	  for(unsigned int j = 0; j < geo->m_NumLines; j++)
	    {
	      line_t line = {{ geo->m_Lines[j*2+0],
			       geo->m_Lines[j*2+1] }};

	      //check if this line is inside the box.  if it is,
	      //add it to the set
	      boost::dynamic_bitset<> inside;
	      for(unsigned int k = 0; k < 2; k++)
		if(zoomed_box.minx <= geometry.points[line[k]][0] &&
		   zoomed_box.miny <= geometry.points[line[k]][1] &&
		   zoomed_box.minz <= geometry.points[line[k]][2] &&
		   zoomed_box.maxx >= geometry.points[line[k]][0] &&
		   zoomed_box.maxx >= geometry.points[line[k]][1] &&
		   zoomed_box.maxx >= geometry.points[line[k]][2])
		  inside.push_back(true);
	      if(inside.size()==3)
		geometry.lines.push_back(line);
	    }
	  for(unsigned int j = 0; j < geo->m_NumTris; j++)
	    {
	      triangle_t tri = {{ geo->m_Tris[j*3+0],
			     geo->m_Tris[j*3+1],
			     geo->m_Tris[j*3+2]  }};

	      //check if this line is inside the box.  if it is,
	      //add it to the set
	      boost::dynamic_bitset<> inside;
	      for(unsigned int k = 0; k < 3; k++)
		if(zoomed_box.minx <= geometry.points[tri[k]][0] &&
		   zoomed_box.miny <= geometry.points[tri[k]][1] &&
		   zoomed_box.minz <= geometry.points[tri[k]][2] &&
		   zoomed_box.maxx >= geometry.points[tri[k]][0] &&
		   zoomed_box.maxx >= geometry.points[tri[k]][1] &&
		   zoomed_box.maxx >= geometry.points[tri[k]][2])
		  inside.push_back(true);
	      if(inside.size()==3)
		geometry.tris.push_back(tri);
	    }
	  for(unsigned int j = 0; j < geo->m_NumQuads; j++)
	    {
	      quad_t quad = {{ geo->m_Quads[j*4+0],
			       geo->m_Quads[j*4+1],
			       geo->m_Quads[j*4+2],
			       geo->m_Quads[j*4+3]  }};

	      //check if this line is inside the box.  if it is,
	      //add it to the set
	      boost::dynamic_bitset<> inside;
	      for(unsigned int k = 0; k < 4; k++)
		if(zoomed_box.minx <= geometry.points[quad[k]][0] &&
		   zoomed_box.miny <= geometry.points[quad[k]][1] &&
		   zoomed_box.minz <= geometry.points[quad[k]][2] &&
		   zoomed_box.maxx >= geometry.points[quad[k]][0] &&
		   zoomed_box.maxx >= geometry.points[quad[k]][1] &&
		   zoomed_box.maxx >= geometry.points[quad[k]][2])
		  inside.push_back(true);
	      if(inside.size()==3)
		geometry.quads.push_back(quad);
	    }
	}

      Geometry newgeo;
      if(geo->m_GeoFrame)
	{
	  newgeo.m_GeoFrame.reset(new LBIE::geoframe);
	  *(newgeo.m_GeoFrame) = *(geo->m_GeoFrame);
	  newgeo.m_GeoFrame->triangles.clear();
	  for(unsigned int j = 0; j < geometry.tris.size(); j++)
	    {
	      LBIE::geoframe::uint_3 tri = {{ geometry.tris[j][0],
					      geometry.tris[j][1],
					      geometry.tris[j][2] }};
	      newgeo.m_GeoFrame->triangles.push_back(tri);
	    }
	  for(unsigned int j = 0; j < geometry.quads.size(); j++)
	    {
	      LBIE::geoframe::uint_4 quad = {{ geometry.quads[j][0],
					       geometry.quads[j][1],
					       geometry.quads[j][2],
					       geometry.quads[j][3] }};
	      newgeo.m_GeoFrame->quads.push_back(quad);
	    }
	}
      else
	{
	  if(!geometry.lines.empty())
	    newgeo.AllocateLines(geometry.points.size(),geometry.lines.size());
	  if(!geometry.tris.empty())
	    newgeo.AllocateTris(geometry.points.size(),geometry.tris.size());
	  if(!geometry.quads.empty())
	    newgeo.AllocateQuads(geometry.points.size(),geometry.quads.size());

	  if(!geometry.colors.empty())
	    newgeo.AllocatePointColors();

	  for(unsigned int j = 0; j < geometry.points.size(); j++)
	    for(unsigned int k = 0; k < 3; k++)
	      newgeo.m_Points[j*3+k] = geometry.points[j][k];
	  
	  for(unsigned int j = 0; j < geometry.lines.size(); j++)
	    for(unsigned int k = 0; k < 2; k++)
	      newgeo.m_Lines[j*2+k] = geometry.lines[j][k];

	  for(unsigned int j = 0; j < geometry.tris.size(); j++)
	    for(unsigned int k = 0; k < 3; k++)
	      newgeo.m_Tris[j*3+k] = geometry.tris[j][k];

	  for(unsigned int j = 0; j < geometry.quads.size(); j++)
	    for(unsigned int k = 0; k < 4; k++)
	      newgeo.m_Quads[j*4+k] = geometry.quads[j][k];
	}

      *geo = newgeo;
    }

  m_ZoomedIn->updateGL();
  m_ZoomedOut->updateGL();
}

void NewVolumeMainWindow::toggleTerminalSlot(bool show)
{
  if(show)
    m_Terminal->show();
  else
    m_Terminal->hide();
}


#ifdef VOLUMEGRIDROVER
#ifdef DETACHED_VOLUMEGRIDROVER
void NewVolumeMainWindow::toggleVolumeGridRoverSlot(bool show)
{
  if(show)
    m_VolumeGridRover->show();
  else
    m_VolumeGridRover->hide();
}
#else
void NewVolumeMainWindow::toggleVolumeGridRoverSlot(bool show)
{
}
#endif
#endif

void NewVolumeMainWindow::virusSegmentationSlot()
{
	// make sure data is loaded
	if (m_SourceManager.hasSource()) {
		SegmentationDialog dialog(this,"SegmentationDialog",true);
		QSettings settings;
		settings.insertSearchPath(QSettings::Windows, "/CCV");

		//set up dialog with saved settings
		dialog.m_SegTypeSelection->setCurrentItem(settings.readNumEntry("/Volume Rover/Segmentation/SegTypeSelection"));
		dialog.m_OptionsStack->raiseWidget(dialog.m_SegTypeSelection->currentItem());
		dialog.m_CapsidLayerType->setCurrentItem(settings.readNumEntry("/Volume Rover/Segmentation/CapsidLayerType"));
		dialog.m_CapsidOptionsStack->raiseWidget(dialog.m_CapsidLayerType->currentItem());
		dialog.m_TLowEditType0->setText(settings.readEntry("/Volume Rover/Segmentation/TLowEditType0"));
		dialog.m_X0EditType0->setText(settings.readEntry("/Volume Rover/Segmentation/X0EditType0"));
		dialog.m_Y0EditType0->setText(settings.readEntry("/Volume Rover/Segmentation/Y0EditType0"));
		dialog.m_Z0EditType0->setText(settings.readEntry("/Volume Rover/Segmentation/Z0EditType0"));
		dialog.m_TLowEditType1->setText(settings.readEntry("/Volume Rover/Segmentation/TLowEditType1"));
		dialog.m_X0EditType1->setText(settings.readEntry("/Volume Rover/Segmentation/X0EditType1"));
		dialog.m_Y0EditType1->setText(settings.readEntry("/Volume Rover/Segmentation/Y0EditType1"));
		dialog.m_Z0EditType1->setText(settings.readEntry("/Volume Rover/Segmentation/Z0EditType1"));
		dialog.m_X1EditType1->setText(settings.readEntry("/Volume Rover/Segmentation/X1EditType1"));
		dialog.m_Y1EditType1->setText(settings.readEntry("/Volume Rover/Segmentation/Y1EditType1"));
		dialog.m_Z1EditType1->setText(settings.readEntry("/Volume Rover/Segmentation/Z1EditType1"));
		dialog.m_TLowEditType2->setText(settings.readEntry("/Volume Rover/Segmentation/TLowEditType2"));
		dialog.m_SmallRadiusEditType2->setText(settings.readEntry("/Volume Rover/Segmentation/SmallRadiusEditType2"));
		dialog.m_LargeRadiusEditType2->setText(settings.readEntry("/Volume Rover/Segmentation/LargeRadiusEditType2"));
		dialog.m_TLowEditType3->setText(settings.readEntry("/Volume Rover/Segmentation/TLowEditType3"));
		dialog.m_3FoldEditType3->setText(settings.readEntry("/Volume Rover/Segmentation/3FoldEditType3"));
		dialog.m_5FoldEditType3->setText(settings.readEntry("/Volume Rover/Segmentation/5FoldEditType3"));
		dialog.m_6FoldEditType3->setText(settings.readEntry("/Volume Rover/Segmentation/6FoldEditType3"));
		dialog.m_SmallRadiusEditType3->setText(settings.readEntry("/Volume Rover/Segmentation/SmallRadiusEditType3"));
		dialog.m_LargeRadiusEditType3->setText(settings.readEntry("/Volume Rover/Segmentation/LargeRadiusEditType3"));
		dialog.m_FoldNumEdit->setText(settings.readEntry("/Volume Rover/Segmentation/FoldNumEdit"));
		dialog.m_HNumEdit->setText(settings.readEntry("/Volume Rover/Segmentation/HNumEdit"));
		dialog.m_KNumEdit->setText(settings.readEntry("/Volume Rover/Segmentation/KNumEdit"));
		dialog.m_3FoldEdit->setText(settings.readEntry("/Volume Rover/Segmentation/3FoldEdit"));
		dialog.m_5FoldEdit->setText(settings.readEntry("/Volume Rover/Segmentation/5FoldEdit"));
		dialog.m_6FoldEdit->setText(settings.readEntry("/Volume Rover/Segmentation/6FoldEdit"));
		dialog.m_InitRadiusEdit->setText(settings.readEntry("/Volume Rover/Segmentation/InitRadiusEdit"));
		dialog.m_HelixWidth->setText(settings.readEntry("/Volume Rover/Segmentation/HelixWidth").isEmpty() ? 
					     dialog.m_HelixWidth->text() : 
					     settings.readEntry("/Volume Rover/Segmentation/HelixWidth"));
		dialog.m_MinHelixWidthRatio->setText(settings.readEntry("/Volume Rover/Segmentation/MinHelixWidthRatio").isEmpty() ?
						     dialog.m_MinHelixWidthRatio->text() : 
						     settings.readEntry("/Volume Rover/Segmentation/MinHelixWidthRatio"));
		dialog.m_MaxHelixWidthRatio->setText(settings.readEntry("/Volume Rover/Segmentation/MaxHelixWidthRatio").isEmpty() ?
						     dialog.m_MaxHelixWidthRatio->text() :
						     settings.readEntry("/Volume Rover/Segmentation/MaxHelixWidthRatio"));
		dialog.m_MinHelixLength->setText(settings.readEntry("/Volume Rover/Segmentation/MinHelixLength").isEmpty() ?
						 dialog.m_MinHelixLength->text() :
						 settings.readEntry("/Volume Rover/Segmentation/MinHelixLength"));
		dialog.m_SheetWidth->setText(settings.readEntry("/Volume Rover/Segmentation/SheetWidth").isEmpty() ?
					     dialog.m_SheetWidth->text() :
					     settings.readEntry("/Volume Rover/Segmentation/SheetWidth"));
		dialog.m_MinSheetWidthRatio->setText(settings.readEntry("/Volume Rover/Segmentation/MinSheetWidthRatio").isEmpty() ?
						     dialog.m_MinSheetWidthRatio->text() :
						     settings.readEntry("/Volume Rover/Segmentation/MinSheetWidthRatio"));
		dialog.m_MaxSheetWidthRatio->setText(settings.readEntry("/Volume Rover/Segmentation/MaxSheetWidthRatio").isEmpty() ?
						     dialog.m_MaxSheetWidthRatio->text() :
						     settings.readEntry("/Volume Rover/Segmentation/MaxSheetWidthRatio"));
		dialog.m_SheetExtend->setText(settings.readEntry("/Volume Rover/Segmentation/SheetExtend").isEmpty() ?
					      dialog.m_SheetExtend->text() :
					      settings.readEntry("/Volume Rover/Segmentation/SheetExtend"));

		dialog.m_RemoteSegmentationFilename->setText(m_CurrentVolumeFilename);

		if (dialog.exec() == QDialog::Accepted) {
		
			XmlRpcValue args;
			
			switch(dialog.m_ExecutionLocation->selectedId())
			{
				case 0:
				{
					if(m_LocalSegThread && m_LocalSegThread->running())
						QMessageBox::critical(this,"Error","Local Segmentation thread already running!",QMessageBox::Ok,Qt::NoButton,Qt::NoButton);
					else
					{
						if(m_LocalSegThread) delete m_LocalSegThread;
						m_LocalSegThread = new LocalSegThread(m_CurrentVolumeFilename.ascii(),&dialog,this);
						m_LocalSegThread->start();
					}
				}
					break;
				case 1:
				{
					if(m_RemoteSegThread && m_RemoteSegThread->running())
						QMessageBox::critical(this,"Error","Remote Segmentation thread already running!",QMessageBox::Ok,Qt::NoButton,Qt::NoButton);
					else
					{
						if(m_RemoteSegThread) delete m_RemoteSegThread;
						m_RemoteSegThread = new RemoteSegThread(&dialog,this);
						m_RemoteSegThread->start();
					}
				}
					break;
			}
						
			// save dialog settings
			settings.writeEntry("/Volume Rover/Segmentation/SegTypeSelection", dialog.m_SegTypeSelection->currentItem());
			settings.writeEntry("/Volume Rover/Segmentation/CapsidLayerType", dialog.m_CapsidLayerType->currentItem());
			settings.writeEntry("/Volume Rover/Segmentation/TLowEditType0", dialog.m_TLowEditType0->text());
			settings.writeEntry("/Volume Rover/Segmentation/X0EditType0", dialog.m_X0EditType0->text());
			settings.writeEntry("/Volume Rover/Segmentation/Y0EditType0", dialog.m_Y0EditType0->text());
			settings.writeEntry("/Volume Rover/Segmentation/Z0EditType0", dialog.m_Z0EditType0->text());
			settings.writeEntry("/Volume Rover/Segmentation/TLowEditType1", dialog.m_TLowEditType1->text());
			settings.writeEntry("/Volume Rover/Segmentation/X0EditType1", dialog.m_X0EditType1->text());
			settings.writeEntry("/Volume Rover/Segmentation/Y0EditType1", dialog.m_Y0EditType1->text());
			settings.writeEntry("/Volume Rover/Segmentation/Z0EditType1", dialog.m_Z0EditType1->text());
			settings.writeEntry("/Volume Rover/Segmentation/X1EditType1", dialog.m_X1EditType1->text());
			settings.writeEntry("/Volume Rover/Segmentation/Y1EditType1", dialog.m_Y1EditType1->text());
			settings.writeEntry("/Volume Rover/Segmentation/Z1EditType1", dialog.m_Z1EditType1->text());
			settings.writeEntry("/Volume Rover/Segmentation/TLowEditType2", dialog.m_TLowEditType2->text());
			settings.writeEntry("/Volume Rover/Segmentation/SmallRadiusEditType2", dialog.m_SmallRadiusEditType2->text());
			settings.writeEntry("/Volume Rover/Segmentation/LargeRadiusEditType2", dialog.m_LargeRadiusEditType2->text());
			settings.writeEntry("/Volume Rover/Segmentation/TLowEditType3", dialog.m_TLowEditType3->text());
			settings.writeEntry("/Volume Rover/Segmentation/3FoldEditType3", dialog.m_3FoldEditType3->text());
			settings.writeEntry("/Volume Rover/Segmentation/5FoldEditType3", dialog.m_5FoldEditType3->text());
			settings.writeEntry("/Volume Rover/Segmentation/6FoldEditType3", dialog.m_6FoldEditType3->text());
			settings.writeEntry("/Volume Rover/Segmentation/SmallRadiusEditType3", dialog.m_SmallRadiusEditType3->text());
			settings.writeEntry("/Volume Rover/Segmentation/LargeRadiusEditType3", dialog.m_LargeRadiusEditType3->text());
			settings.writeEntry("/Volume Rover/Segmentation/FoldNumEdit", dialog.m_FoldNumEdit->text());
			settings.writeEntry("/Volume Rover/Segmentation/HNumEdit", dialog.m_HNumEdit->text());
			settings.writeEntry("/Volume Rover/Segmentation/KNumEdit", dialog.m_KNumEdit->text());
			settings.writeEntry("/Volume Rover/Segmentation/3FoldEdit", dialog.m_3FoldEdit->text());
			settings.writeEntry("/Volume Rover/Segmentation/5FoldEdit", dialog.m_5FoldEdit->text());
			settings.writeEntry("/Volume Rover/Segmentation/6FoldEdit", dialog.m_6FoldEdit->text());
			settings.writeEntry("/Volume Rover/Segmentation/InitRadiusEdit", dialog.m_InitRadiusEdit->text());
			settings.writeEntry("/Volume Rover/Segmentation/HelixWidth", dialog.m_HelixWidth->text());
			settings.writeEntry("/Volume Rover/Segmentation/MinHelixWidthRatio", dialog.m_MinHelixWidthRatio->text());
			settings.writeEntry("/Volume Rover/Segmentation/MaxHelixWidthRatio", dialog.m_MaxHelixWidthRatio->text());
			settings.writeEntry("/Volume Rover/Segmentation/MinHelixLength", dialog.m_MinHelixLength->text());
			settings.writeEntry("/Volume Rover/Segmentation/SheetWidth", dialog.m_SheetWidth->text());
			settings.writeEntry("/Volume Rover/Segmentation/MinSheetWidthRatio", dialog.m_MinSheetWidthRatio->text());
			settings.writeEntry("/Volume Rover/Segmentation/MaxSheetWidthRatio", dialog.m_MaxSheetWidthRatio->text());
			settings.writeEntry("/Volume Rover/Segmentation/SheetExtend", dialog.m_SheetExtend->text());
		}
	}
	else {
		QMessageBox::warning(this, tr("Segment Virus Map"),
			tr("A volume must be loaded for this feature to work."), 0,0);
	}
}

#define IndexVect(i,j,k) ((k)*xdim*ydim + (j)*xdim + (i))
#ifndef _max_
#define _max_(x, y) ((x>y) ? (x):(y))
#endif
#ifndef _min_
#define _min_(x, y) ((x<y) ? (x):(y))
#endif

void NewVolumeMainWindow::contrastEnhancementSlot()
{
  if(m_SourceManager.hasSource())
    {
      if(!m_RGBARendering)
	{
	  ContrastEnhancementDialog dialog(this);

	  dialog.m_Resistor->setValidator(new QDoubleValidator(this));

	  if(dialog.exec() == QDialog::Accepted)
	    {
	      if(dialog.m_Resistor->text().toFloat() < 0.0 ||
		 dialog.m_Resistor->text().toFloat() > 1.0)
		{
		  QMessageBox::warning(this, tr("Contrast Enhancement"),
				       tr("Resistor value must be between 0.0 and 1.0."), 0,0);
		  return;
		}

	      /*
		Check if we are doing in place filtering of the actual volume data file
		instead of simply the current subvolume buffer
		TODO: need to implement out-of-core filtering using VolMagick
	      */
	      if(!dialog.m_Preview->isChecked())
		{
		  if(QMessageBox::warning(this,
					  "Contrast Enhancement",
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
		  vol.contrastEnhancement(dialog.m_Resistor->text().toDouble());
		  VolMagick::writeVolumeFile(vol,
					     m_VolumeFileInfo.filename(),
					     getVarNum(),
					     getTimeStep());
		  QString tmpp(m_VolumeFileInfo.filename().c_str());
		  openFile(tmpp);
		  return;
		}

	      VolumeBuffer *densityBuffer;
	      unsigned int xdim, ydim, zdim, i, len;
	      unsigned char *volume;
	      float *dataset, resistor = dialog.m_Resistor->text().toFloat();
	      unsigned char maxraw, minraw;

	      densityBuffer = m_ZoomedInRenderable.getVolumeBufferManager()->getVolumeBuffer(getVarNum());
	      volume = (unsigned char *)densityBuffer->getBuffer();
	      xdim = densityBuffer->getWidth();
	      ydim = densityBuffer->getHeight();
	      zdim = densityBuffer->getDepth();
	      dataset = new float[xdim*ydim*zdim];

	      m_SubVolumeIsFiltered = true;
	     
	      Q3ProgressDialog progressDialog("Performing contrast enhancement of the sub-volume.","Cancel",zdim*4,this,"Contrast Enhancement", true);
	      progressDialog.setProgress(0);

	      /* load the volume data into the float array */
	      maxraw = 0;
	      minraw = 255;
	      len = xdim*ydim*zdim;
	      for(i=0; i<len; i++)
		{
		  dataset[i] = (float)volume[i];
		  if(volume[i]>maxraw) maxraw = volume[i];
		  if(volume[i]<minraw) minraw = volume[i];
		}
	      qDebug("minimum = %f, maximum = %f\n",(float)minraw,(float)maxraw);
	      
	      /* scale it to span the range of 0-255 */
	      for(i=0; i<len; i++)
		dataset[i] = 255.0*((dataset[i]-minraw)/float(maxraw-minraw));

	      /* now actually perform contrast enhancement */
	      {
		int i,j,k;
		float *upmax, *upmin;
		float *downmax, *downmin;
		float lmin, lmax, img, avg;
		float tempmax,tempmin,temp;
		float window;
		float a,b,c,alpha;
		float *imgavg;

		/* Initialization */
		upmin = new float[xdim*ydim*zdim];
		upmax = new float[xdim*ydim*zdim];
		downmin = new float[xdim*ydim*zdim];
		downmax = new float[xdim*ydim*zdim];
		imgavg = new float[xdim*ydim*zdim];

		memcpy(upmin,dataset,sizeof(float)*len);
		memcpy(upmax,dataset,sizeof(float)*len);
		memcpy(downmin,dataset,sizeof(float)*len);
		memcpy(downmax,dataset,sizeof(float)*len);
		memcpy(imgavg,dataset,sizeof(float)*len);

		/* Bottom-up propagation */
		Filters::ContrastEnhancementSlice(xdim,ydim,resistor,upmin,upmax,imgavg,0);
		progressDialog.setProgress(progressDialog.progress()+1);

		for (k=1; k<(int)zdim; k++) 
		  {
		    /* propagation from lower slice */
		    for (j=0; j<(int)ydim; j++)
		      for (i=0; i<(int)xdim; i++)
			{
			  imgavg[IndexVect(i,j,k)] += resistor*
			    (imgavg[IndexVect(i,j,k-1)]-imgavg[IndexVect(i,j,k)]);
			  if (upmin[IndexVect(i,j,k-1)] < upmin[IndexVect(i,j,k)])
			    upmin[IndexVect(i,j,k)] += resistor*
			      (upmin[IndexVect(i,j,k-1)] - upmin[IndexVect(i,j,k)]);
			  if (upmax[IndexVect(i,j,k-1)] > upmax[IndexVect(i,j,k)])
			    upmax[IndexVect(i,j,k)] += resistor*
			      (upmax[IndexVect(i,j,k-1)] - upmax[IndexVect(i,j,k)]);
			}
		    Filters::ContrastEnhancementSlice(xdim,ydim,resistor,upmin,upmax,imgavg,k);
		    progressDialog.setProgress(progressDialog.progress()+1);
		    qApp->processEvents();
		    if (progressDialog.wasCancelled()) 
		      {
			// the filtering did not complete
			m_SubVolumeIsFiltered = false;
			// let the buffer manager know what happened
			m_ZoomedInRenderable.getVolumeBufferManager()->markBufferAsInvalid(getVarNum());
			
			/* clean up */
			delete upmin;
			delete upmax;
			delete downmin;
			delete downmax;
			delete imgavg;
			delete dataset;

			return;
		      }
		  }
		
		/* Top-down propagation */
		Filters::ContrastEnhancementSlice(xdim,ydim,resistor,downmin,downmax,imgavg,zdim-1);
		progressDialog.setProgress(progressDialog.progress()+1);

		for (k=zdim-2; k>=0; k--) 
		  {
		    /* propagation from upper slice */
		    for (j=0; j<(int)ydim; j++)
		      for (i=0; i<(int)xdim; i++) 
			{
			  imgavg[IndexVect(i,j,k)] += resistor*
			    (imgavg[IndexVect(i,j,k+1)]-imgavg[IndexVect(i,j,k)]);
			  if (downmin[IndexVect(i,j,k+1)] < downmin[IndexVect(i,j,k)])
			    downmin[IndexVect(i,j,k)] += resistor*
			      (downmin[IndexVect(i,j,k+1)] - downmin[IndexVect(i,j,k)]);
			  if (downmax[IndexVect(i,j,k+1)] > downmax[IndexVect(i,j,k)])
			    downmax[IndexVect(i,j,k)] += resistor*
			      (downmax[IndexVect(i,j,k+1)] - downmax[IndexVect(i,j,k)]);
			}
		  
		    Filters::ContrastEnhancementSlice(xdim,ydim,resistor,downmin,downmax,imgavg,k);
		    progressDialog.setProgress(progressDialog.progress()+1);
		    qApp->processEvents();
		    if (progressDialog.wasCancelled()) 
		      {
			// the filtering did not complete
			m_SubVolumeIsFiltered = false;
			// let the buffer manager know what happened
			m_ZoomedInRenderable.getVolumeBufferManager()->markBufferAsInvalid(getVarNum());
			
			/* clean up */
			delete upmin;
			delete upmax;
			delete downmin;
			delete downmax;
			delete imgavg;
			delete dataset;

			return;
		      }
		  }
		
		/* Stretching */
		tempmin = 9999;
		tempmax = -9999;
		for (k=0; k<(int)zdim; k++)
		  {
		    for (j=0; j<(int)ydim; j++)
		      for (i=0; i<(int)xdim; i++)
			{
			  
			  lmin = _min_(upmin[IndexVect(i,j,k)], downmin[IndexVect(i,j,k)]);
			  lmax = _max_(upmax[IndexVect(i,j,k)], downmax[IndexVect(i,j,k)]);
			  img = dataset[IndexVect(i,j,k)];
			  avg = imgavg[IndexVect(i,j,k)];
			  
			  window = lmax - lmin;
			  window = sqrt(window*(510-window));
			  
			  
			  if (lmin != lmax) 
			    {
			      img = window*(img-lmin)/(lmax-lmin);
			      avg = window*(avg-lmin)/(lmax-lmin);
			    }
			  
			  alpha = (avg-img)/(181.019 * window);
			  if (alpha != 0) 
			    {
			    
			      a = 0.707 * alpha;
			      b = 1.414*alpha*(img - window) - 1;
			      c = 0.707*alpha*img*(img-2*window) + img;
			      
			      imgavg[IndexVect(i,j,k)] = lmin+(-b-sqrt(b*b-4*a*c))/(2*a);
			      
			    }
			  else 
			    {
			  
			      imgavg[IndexVect(i,j,k)] = img + lmin;
			      
			    }
			  
			  temp = imgavg[IndexVect(i,j,k)];
			  if (temp > tempmax)
			    tempmax = temp;
			  if (temp < tempmin)
			    tempmin = temp;
			  
			}
		    
		    progressDialog.setProgress(progressDialog.progress()+1);
		    qApp->processEvents();
		    if (progressDialog.wasCancelled()) 
		      {
			// the filtering did not complete
			m_SubVolumeIsFiltered = false;
			// let the buffer manager know what happened
			m_ZoomedInRenderable.getVolumeBufferManager()->markBufferAsInvalid(getVarNum());
			
			/* clean up */
			delete upmin;
			delete upmax;
			delete downmin;
			delete downmax;
			delete imgavg;
			delete dataset;

			return;
		      }
		  }
   
		for (k=0; k<(int)zdim; k++)
		  {
		    for (j=0; j<(int)ydim; j++)
		      for (i=0; i<(int)xdim; i++)
			{
			  if (imgavg[IndexVect(i,j,k)] < tempmin)
			    imgavg[IndexVect(i,j,k)] = tempmin;
			  if (imgavg[IndexVect(i,j,k)] > tempmax)
			    imgavg[IndexVect(i,j,k)] = tempmax;
			  
			  dataset[IndexVect(i,j,k)] = (float)((imgavg[IndexVect(i,j,k)]-tempmin)
							      *255.0/(tempmax-tempmin));
			}
		    
		    progressDialog.setProgress(progressDialog.progress()+1);
		    qApp->processEvents();
		    if (progressDialog.wasCancelled()) 
		      {
			// the filtering did not complete
			m_SubVolumeIsFiltered = false;
			// let the buffer manager know what happened
			m_ZoomedInRenderable.getVolumeBufferManager()->markBufferAsInvalid(getVarNum());
			
			/* clean up */
			delete upmin;
			delete upmax;
			delete downmin;
			delete downmax;
			delete imgavg;
			delete dataset;

			return;
		      }
		  }

		/* copy the new data to the volume buffer */
		for(i=0; i<(int)len; i++)
		  volume[i] = (char)dataset[i];

		/* clean up */
		delete upmin;
		delete upmax;
		delete downmin;
		delete downmax;
		delete imgavg;
		delete dataset;
	      }

	      if(m_SubVolumeIsFiltered)
		{
		  m_ZoomedIn->makeCurrent();
		  updateRoverRenderable(&m_ZoomedInRenderable, &m_ZoomedInExtents);
printf("NewVolumeMainWindow::contrastEnhancementSlot()\n");
		  m_ZoomedIn->updateGL();
		}
	    }
	}
      else
	QMessageBox::warning(this, tr("Contrast Enhancement"),
			     tr("This feature is not available for RGBA volumes."), 0,0);
    }
  else
    QMessageBox::warning(this, tr("Contrast Enhancement"),
			 tr("A volume must be loaded for this feature to work."), 0,0);
}

#ifdef USING_PE_DETECTION

extern cVesselSeg<unsigned char> VesselSeg_g;

void PEDetection(const char *filename);

void NewVolumeMainWindow::PEDetectionSlot()
{
  if(m_SourceManager.hasSource())
    {
      PEDetectionDialog dialog(this,"PEDetectionDialog",true);

      if(!m_CurrentVolumeFilename.endsWith(".rawiv"))
	{
	  QMessageBox::critical(this, tr("Pulmonary Embolus Detection"),
				tr("Only RawIV files are supported."), 0,0);
	  return;
	}
      
      dialog.m_Port->setValidator(new QIntValidator(dialog.m_Port));
      dialog.m_RemoteFile->setText(m_CurrentVolumeFilename);

      if(dialog.exec() == QDialog::Accepted)
	{
	  if(m_PEDetectionThread && m_PEDetectionThread->running())
	    QMessageBox::critical(this,"Error","Pulmonary Embolus thread already running!");
	  else
	    {
	      if(m_PEDetectionThread) delete m_PEDetectionThread;
	      m_PEDetectionThread = new PEDetectionThread(m_CurrentVolumeFilename.ascii(),
							  &dialog,
							  this);
	      m_PEDetectionThread->start();
	    }
	}
    }
  else
    QMessageBox::critical(this, tr("Pulmonary Embolus Detection"),
			  tr("A volume must be loaded for this feature to work."), 0,0);
}
#else
void NewVolumeMainWindow::PEDetectionSlot()
{
  QMessageBox::information(this, tr("Pulmonary Embolus Detection"),
			   tr("PE Detection disabled"), QMessageBox::Ok);
}
#endif

void NewVolumeMainWindow::saveSubvolumeSlot()
{
	// save the zoomed in volume to a file
	// perhaps it would be best to put this somewhere else?
	// (VolumeBufferManager? SourceManager?)

	// doesn't work for now
	//qDebug("NewVolumeMainWindow::saveSubvolumeSlot()");


	if (m_SourceManager.hasSource()) {
		// prompt for a filename (and format just to be sure)
		QString fileName, format;
	 
		if (m_RGBARendering) // multivariable
			fileName = Q3FileDialog::getSaveFileName( QString::null, "RawV files (*.rawv)", 
			this, "save file dialog", "Choose a filename", &format);
		else {
			// single variable
			fileName = Q3FileDialog::getSaveFileName( QString::null,
					"Rawiv files (*.rawiv);;RawV files (*.rawv);;MRC files (*.mrc)",
					this, "save file dialog", "Choose a filename", &format);
		}

		if(fileName.isEmpty()) return;

		//qDebug("format selected: %s; filename: %s", format.ascii(),
		//				fileName.ascii());
		// pick the extension out of the format string
		QString extension = format.section('.', -1, -1);
		// chop off the trailing )
		extension = extension.left(extension.length()-1);
		//qDebug("selected extension: %s", extension.ascii());
		
		// check the filename's extension against the format string's extension
		QFileInfo fileInfo(fileName);

		if (!fileInfo.extension(false).isNull()
				&& fileInfo.extension(false) != extension) {
			QMessageBox::warning(this, tr("Save Subvolume"),
					tr("The filename's extension does not match the selected file type\n"
						 "(the file will still be saved)"),
				 	0,0);
		}

		// save it!
		if (!fileName.isNull()) {
			VolumeFileSink* fileSink = new VolumeFileSink(fileName, extension);
			VolumeTranscriber volTrans(m_SourceManager.getSource(), fileSink);
			int startX,startY,startZ,endX,endY,endZ;
			double minX,minY,minZ, maxX,maxY,maxZ;
			bool goResult = true;


			// copy the subvolume's extents
			minX = m_ZoomedInExtents.getXMin();
			minY = m_ZoomedInExtents.getYMin();
			minZ = m_ZoomedInExtents.getZMin();
			maxX = m_ZoomedInExtents.getXMax();
			maxY = m_ZoomedInExtents.getYMax();
			maxZ = m_ZoomedInExtents.getZMax();
			
			// get the coordinates of the subvolume
			if (m_ZoomedInRenderable.getVolumeBufferManager()->
					getDownLoadManager()->getExtentCoords(&m_SourceManager,
									minX,minY,minZ, maxX,maxY,maxZ,
									startX,startY,startZ, endX,endY,endZ)) {
				qDebug("saving subvolume from %d,%d,%d,  to %d,%d,%d",
									startX,startY,startZ, endX,endY,endZ);

				//bool go(uint minX, uint minY, uint minZ, uint minT,
				//						uint maxX, uint maxY, uint maxZ, uint maxT);
				// if (m_SubVolumeIsFiltered)
// 					goResult = volTrans.goFiltered(this, startX,startY,startZ,
// 											getTimeStep(), endX,endY,endZ, getTimeStep());
// 				else
					goResult = volTrans.go(this, startX,startY,startZ, getTimeStep(),
													 endX,endY,endZ, getTimeStep());

				if (!goResult) {
					// an error occurred
					QMessageBox::warning(this, tr("Save Subvolume"),
						tr("An error occurred"), 0,0);
				}
			}

			// clean up
			delete fileSink;
				
			if (!goResult) {
				// the transcription failed
				// remove the file used by the VolumeFileSink


			  // arand: commended three lines below for Qt4 complication
			  cout << "WARNING: Reached some code that is likely broken." << endl;
			  //QFileInfo fileInfo(fileName);
			  //QDir fileDir = fileInfo.dir();	
			  //fileDir.remove(fileInfo.absFilePath(), true);
			}
		}
	}
	else {
		QMessageBox::warning(this, tr("Save Subvolume"),
			tr("A volume must be loaded for this feature to work."), 0,0);
	}
	//return;
}

void NewVolumeMainWindow::saveImageSlot()
{
	ImageSaveDialog dialog(this, 0, true);
	
	if (dialog.exec() == QDialog::Accepted) {
		// which format are we using?
		QString format = dialog.imageFormatMenu->currentText();
		//qDebug("format: %s", (const char *)format);
		
		// where should we save it?
		QString fileName =
			Q3FileDialog::getSaveFileName("", QString("*.%1").arg(format.lower()), this );
		//qDebug("filename: %s", (const char *)fileName);
		
		if (!fileName.isNull()) {
			// which radio button is checked?
			if (dialog.subVolumeButton->isChecked()){
				//qDebug("saving the subvolume render");
				// redraw
				m_ZoomedIn->updateGL();
				// save the image
				QImage saveImg = m_ZoomedIn->grabFrameBuffer();
				saveImg.save(fileName, (const char *)format);
			}
			else if (dialog.thumbVolumeButton->isChecked()) {
				//qDebug("saving the thumbnail volume render");
				// redraw
				m_ZoomedOut->updateGL();
				// save the image
				QImage saveImg = m_ZoomedOut->grabFrameBuffer();
				saveImg.save(fileName, (const char *)format);
			}
			else {
				//qDebug("saving both renders");
			}
		}
	}
}

// XXX
void NewVolumeMainWindow::startRecordingAnimationSlot()
{
	if (!m_AnimationRecording && !m_AnimationPlaying) {
		qDebug("Begin recording animation...");
		// we're recording an animation
		m_AnimationRecording = true;

		// delete any previous animation
		if (m_Animation) delete m_Animation;

		// create the animation object
		ViewInformation info = m_ZoomedOut->getViewInformation();
		info.setClipPlane((float)m_ThumbnailRenderable.getVolumeRenderer()->getNearPlane());
		m_Animation = new Animation(ViewState(info.getOrientation(),
														info.getTarget(), info.getWindowSize(),
														info.getClipPlane(), m_WireFrame));

		// start the animation sample timer
		m_AnimationTimerId = startTimer(c_SampleInterval);
		// start the animation timer
		m_Time.start();
	}
}

void NewVolumeMainWindow::stopRecordingAnimationSlot()
{
	if (m_AnimationRecording) {
		qDebug("Stop recording animation");
		// record one final frame
		recordFrame();
		// we're no longer recording an animation
		m_AnimationRecording = false;
		// stop the sample timer
		killTimer(m_AnimationTimerId);
		// don't touch the animation object... it stays around for playback
		// or saving
	}
}

void NewVolumeMainWindow::playAnimationSlot()
{
	if (!m_AnimationRecording && !m_AnimationPlaying && m_Animation) {
		qDebug("Begin Playing Animation...");
		// we're playing back an animation
		m_AnimationPlaying = true;

		// start the event timer (no delay... play as many frames as possible)
		m_AnimationTimerId = startTimer(0);
		// start the time object
		m_Time.start();
	}
}

void NewVolumeMainWindow::stopAnimationSlot()
{
	if (m_AnimationPlaying) {
		qDebug("Stop Playing Animation...");
		// we're no longer playing the animation
		m_AnimationPlaying = false;
		// stop the timer
		killTimer(m_AnimationTimerId);
		m_AnimationTimerId = 0;
		// clear the frame writing fields

		m_FrameNumber = 0;

		m_SaveAnimationFrame = false;

		m_AnimationFrameName = QString::null;

		// sync the views
		mouseReleasedMain();
	}
}

void NewVolumeMainWindow::saveAnimationSlot()
{
	// make sure the animation exist and is not currently being recorded
	if (m_Animation && !m_AnimationRecording) {
		qDebug("Save Animation...");

		FILE *fp = NULL;
		QString filename = Q3FileDialog::getSaveFileName(QString::null, "Text Files (*.txt)", this);
		
		if (!filename.isNull()) {
			// open the file
			fp = fopen(filename.ascii(), "w");

			if (fp) {
				// write it
				m_Animation->writeAnimation(fp);
			}
			else {
				QMessageBox::information(this, "Error Creating File", "There was an error creating the file");
			}
		}
	}
	else {
		QMessageBox::information(this, "Error Saving Animation", "The animation failed to save because one has not been created or is still being recorded.");
	}
}

void NewVolumeMainWindow::loadAnimationSlot()
{
	if (!m_AnimationRecording && !m_AnimationPlaying) {
		qDebug("Load Animation...");

		if (m_Animation) {
			// XXX: warn about overwriting animation
			int result = QMessageBox::warning(this, "Overwrite Warning", "If you load an animation, the current animation will be discarded. Continue?", QMessageBox::Yes, QMessageBox::No);

			// bail if the answer was no
			if (result == QMessageBox::No)
				return;
		}
		
		FILE *fp = NULL;
		QString filename = Q3FileDialog::getOpenFileName(QString::null, "Text Files (*.txt)", this);

		if (!filename.isNull()) {
			// open the file
			fp = fopen(filename.ascii(), "r");

			if (fp) {
				// create an animation if needed
				if (!m_Animation) {
					ViewInformation info = m_ZoomedOut->getViewInformation();
					m_Animation = new Animation(ViewState(info.getOrientation(),
														info.getTarget(), info.getWindowSize(),
														0.0, m_WireFrame));
				}
				// read it
				m_Animation->readAnimation(fp);


				// close the file

				fclose(fp);

			}
			else {
				QMessageBox::information(this, "Error Opening File", "There was an error opening the file");
			}
		}
	}
	else {
		// XXX: can't load during record or play
		QMessageBox::information(this, "Error Loading Animation", "You must stop recording or playback of the current animation before loading an animation");
	}
}

void NewVolumeMainWindow::renderSequenceSlot()
{
	// prompt for a filename
	QString filename = Q3FileDialog::getSaveFileName(QString::null, "All Files (*.*)", this);
	
	if (!filename.isNull()) {
		// start playback
		playAnimationSlot();

		// set the frame filename
		m_AnimationFrameName = filename;

		// set the save frame flag if playback started successfully
		if (m_AnimationPlaying)
			m_SaveAnimationFrame = true;
	}
}



unsigned int NewVolumeMainWindow::getVarNum() const
{
	return m_VariableBox->currentItem();
}

unsigned int NewVolumeMainWindow::getTimeStep() const
{
	if (m_RGBARendering)
		return m_RGBATimeStep->value();
	else 
		return m_TimeStep->value();
}

void NewVolumeMainWindow::loadGeometrySlot()
{
	/*
	QString filename = QFileDialog::getOpenFileName(QString::null, "IPoly files (*.ipoly)", this);

	if (!filename.isNull()) {
		IPolyRenderable* geometry = new IPolyRenderable;
		if (geometry->loadFile(filename)) {
			m_SourceManager.addGeometry(geometry);
			m_ZoomedIn->updateGL();
			m_ZoomedOut->updateGL();
		}
		else {
			delete geometry;
		}
	}
	
	QString filename = QFileDialog::getOpenFileName(QString::null, "rawn files (*.rawn)", this);

	if (!filename.isNull()) {
		Geometry* geometry = loadGeometry(filename);
		if (geometry) {
			GeometryRenderable* geometryRenderable = new GeometryRenderable(geometry);
			m_SourceManager.addGeometry(geometryRenderable);
			m_ZoomedIn->updateGL();
			m_ZoomedOut->updateGL();
		}
		else {
			QMessageBox::information(this, "Error Opening File", "There was an error opening the file");
		}
	}*/
  /*
	GeometryLoader loader;

	QString filename = QFileDialog::getOpenFileName(QString::null, (QString)((loader.getLoadFilterString()+";;All Files (*.*)").c_str()), this);
	if (!filename.isNull()) {
		Geometry* geometry = loader.loadFile((std::string)(filename.ascii()));
		if (geometry) {
			GeometryRenderable* geometryRenderable = new GeometryRenderable(geometry);
			m_Geometries.add(geometryRenderable);
			m_ZoomedIn->updateGL();
			m_ZoomedOut->updateGL();
		}
		else {
			QMessageBox::information(this, "Error Opening File", "There was an error opening the file");
		}
	}
  */

  GeometryLoader loader;
  Geometry* geometry;

  qDebug("loader.getLoadFilterString(): %s", loader.getLoadFilterString().c_str());
  QStringList filenames = Q3FileDialog::getOpenFileNames((QString)((loader.getLoadFilterString()+";;All Files (*)").c_str()),
							QString::null,
							this,
							"Open Geometry Files",
							"Select one or more Geometry files to open");
  for(QStringList::Iterator it = filenames.begin(); it != filenames.end(); it++)
    {
#if 0
      //for now, first try loading it as an LBIE mesh using LBIE::geoframe::read_raw
      //TODO: add mesh reading from read_raw() to GeometryFileTypes
      LBIE::geoframe g_frame;
      if(g_frame.read_raw((*it).ascii()) != -1)
	{
	  qDebug("Loading %s with LBIE::geoframe::read_raw",(*it).ascii());
	  //g_frame.calculatenormals();

	  //success! now add this geoframe to a Geometry object and put it in the list
	  if(g_frame.mesh_type == 0) //triangle surface rendering of geoframe seems broken
	    {
	      geometry = new Geometry();
	      LBIE::copyGeoframeToGeometry(g_frame,*geometry);
	      GeometryRenderable *gr = new GeometryRenderable(geometry);
	      gr->setWireframeMode(WireframeRenderToggleAction->isOn());
	      gr->setSurfWithWire(RenderSurfaceWithWireframeAction->isOn());
	      m_Geometries.add(gr);
	    }
	  else
	    {
	      //change colors to something better :o
	      for(int i = 0; i < g_frame.color.size(); i++)
		{
		  g_frame.color[i][0] = 0.0;
		  g_frame.color[i][1] = 0.4;
		  g_frame.color[i][2] = 0.2;
		}

	      geometry = new Geometry();
	      geometry->m_GeoFrame.reset(new LBIE::geoframe(g_frame));
	      GeometryRenderable *gr = new GeometryRenderable(geometry);
	      gr->setWireframeMode(WireframeRenderToggleAction->isOn());
	      gr->setSurfWithWire(RenderSurfaceWithWireframeAction->isOn());
	      m_Geometries.add(gr);
	    }
	  continue;
	}

      geometry = loader.loadFile(std::string((*it).ascii()));
      if(geometry)
	{
	  GeometryRenderable* gr = new GeometryRenderable(geometry);
	  gr->setWireframeMode(WireframeRenderToggleAction->isOn());
	  gr->setSurfWithWire(RenderSurfaceWithWireframeAction->isOn());
	  m_Geometries.add(gr);
	}
      else
	QMessageBox::information(this, "Error Opening File", QString("There was an error opening the file '%1'").arg(*it));
#endif

#ifdef USING_GEOMETRY_LOADER
      geometry = loader.loadFile(std::string((*it).ascii()));
      if(geometry)
	{
	  GeometryRenderable* gr = new GeometryRenderable(geometry);
	  gr->setWireframeMode(WireframeRenderToggleAction->isOn());
	  gr->setSurfWithWire(RenderSurfaceWithWireframeAction->isOn());
	  m_Geometries.add(gr);
	}
      else
	QMessageBox::information(this, "Error Opening File", QString("There was an error opening the file '%1'").arg(*it));
#else
      cvcraw_geometry::geometry_t geom;
      cvcraw_geometry::read(geom,(*it).ascii());
      geometry = new Geometry;
      *geometry = Geometry::conv(geom);
      GeometryRenderable *gr = new GeometryRenderable(geometry);
      gr->setWireframeMode(WireframeRenderToggleAction->isOn());
      gr->setSurfWithWire(RenderSurfaceWithWireframeAction->isOn());
      m_Geometries.add(gr);
#endif
    }

  if(!m_SourceManager.hasSource())
    {
      using namespace std;
      using namespace boost;

      //No volume loaded, so lets make an empty volume with a bounding box that can fit all loaded geometry
      qDebug("No Volume Loaded! (Re)Generating empty volume...");

      try
	{
	  VolMagick::BoundingBox globalbox;
	  Geometry *geo;
	  for(unsigned int i=0; i<m_Geometries.getNumberOfRenderables(); i++)
	    {
	      geo = static_cast<GeometryRenderable*>(m_Geometries.get(i))->getGeometry();
	      /*
	      if(geo->m_GeoFrame)
		{
		  geo->m_GeoFrame->calculateExtents();
		  if(i==0)
		    globalbox = VolMagick::BoundingBox(geo->m_GeoFrame->min_x,
						       geo->m_GeoFrame->min_y,
						       geo->m_GeoFrame->min_z,
						       geo->m_GeoFrame->max_x,
						       geo->m_GeoFrame->max_y,
						       geo->m_GeoFrame->max_z);
		  else
		    {
		      globalbox += VolMagick::BoundingBox(geo->m_GeoFrame->min_x,
							  geo->m_GeoFrame->min_y,
							  geo->m_GeoFrame->min_z,
							  geo->m_GeoFrame->max_x,
							  geo->m_GeoFrame->max_y,
							  geo->m_GeoFrame->max_z);
		    }
		}
	      else
	      */
		{
	      
		  geo->GetReadyToDrawWire(); //calculate extents if needed
		  if(i==0)
		    globalbox = VolMagick::BoundingBox(geo->m_Min[0],geo->m_Min[1],geo->m_Min[2],
						       geo->m_Max[0],geo->m_Max[1],geo->m_Max[2]);
		  else
		    globalbox += VolMagick::BoundingBox(geo->m_Min[0],geo->m_Min[1],geo->m_Min[2],
							geo->m_Max[0],geo->m_Max[1],geo->m_Max[2]);
		}
	    }

	  qDebug("%s",str(format("BoundingBox(%1%,%2%,%3%,%4%,%5%,%6%)")
			  % globalbox.minx % globalbox.miny % globalbox.minz
			  % globalbox.maxx % globalbox.maxy % globalbox.maxz).c_str());

	  if(globalbox.isNull())
	    {
	      qDebug("Null bounding box for loaded geometry..something is wrong!");
	      return;
	    }

	  VolMagick::Volume empty_vol;
	  empty_vol.boundingBox(globalbox);
	  QDir tmpdir(m_CacheDir.absPath() + "/tmp");
	  if(!tmpdir.exists()) tmpdir.mkdir(tmpdir.absPath());
	  QFileInfo tmpfile(tmpdir,"tmp.rawiv");
	  qDebug("Creating volume %s",tmpfile.absFilePath().ascii());
	  QString newfilename = QDir::convertSeparators(tmpfile.absFilePath());
	  VolMagick::createVolumeFile(newfilename.toStdString(),
				      empty_vol.boundingBox(),
				      empty_vol.dimension(),
				      std::vector<VolMagick::VoxelType>(1, empty_vol.voxelType()));
	  VolMagick::writeVolumeFile(empty_vol,newfilename.toStdString());
	  openFile(newfilename);
	}
      catch(const VolMagick::Exception& e)
	{
	  QMessageBox::critical(this,"Error opening the file",e.what());
	  return;
	}
    }

  m_ZoomedIn->updateGL();
  m_ZoomedOut->updateGL();
}

void NewVolumeMainWindow::clearGeometrySlot()
{
#ifdef USING_SECONDARYSTRUCTURES
	if(ssData) ssData->clearData();
			
#endif

	// get rid of the geometry
	m_Geometries.clear();
	// clear any transformations
	m_ThumbnailRenderable.getGeometryRenderer()->clearTransformation();

	// redraw the windows
	m_ZoomedIn->updateGL();
	m_ZoomedOut->updateGL();
}

void NewVolumeMainWindow::saveGeometrySlot()
{
  GeometryLoader loader;
  QString filter;
  QString filename;


  if(m_Geometries.getNumberOfRenderables()>0) //if we have geometry loaded
    filename = Q3FileDialog::getSaveFileName(QString::null, 
					      (QString)(loader.getSaveFilterString().c_str()), 
					      this, "Save As", "Save As", &filter);
  else
    {
      QMessageBox::warning(this, tr("Save Geometry"),
			   tr("No geometry loaded."), 0, 0);
      return;
    }

  if(filename.isNull())
    {
      //QMessageBox::warning(this, tr("Save Geometry"),
      //tr("Invalid filename."), 0, 0);
      return;
    }

  if(m_Geometries.getNumberOfRenderables() == 1)
    {
      Geometry *geo =  static_cast<GeometryRenderable*>(m_Geometries.get(0))->getGeometry();
      /*
      if(geo->m_GeoFrame)
	{
	  QString typestring("_tri");
	  switch(geo->m_GeoFrame->mesh_type)
	    {
	    case 0: typestring = "_tri"; break;
	    case 1: typestring = "_tet"; break;
	    case 2: typestring = "_quad"; break;
	    case 3: typestring = "_hex"; break;
	    case 4: typestring = "_nurbs"; break;
	    }

	  QString extension = QFileInfo(filename).extension();
	  QString dirpath = QFileInfo(filename).dirPath(true);
	  QString newFileName = QString("%1/%2%3.%4")
	    .arg(dirpath)
	    .arg(QFileInfo(filename).baseName(true))
	    .arg(typestring)
	    .arg(extension);

	  qDebug("Writing geometry using geoframe::write_raw()");
	  geo->m_GeoFrame->write_raw(newFileName.ascii());
	}
      */
      if(!geo->m_SceneGeometry.geometry.empty())
	{
	  cvcraw_geometry::write(geo->m_SceneGeometry.geometry,filename.ascii());
	}
      else
	{
	  if(!loader.saveFile(std::string(filename.ascii()), std::string(filter.ascii()),geo))
	    QMessageBox::information(this, "Error Saving File", "There was an error saving the file");
	}
    }
  else //more than 1 geometry, so save the set of geometries with numbered filenames
    {
      QString extension;
      
      extension = QFileInfo(filename).extension();
      if(extension.isEmpty()) extension = "rawc";

      for(unsigned int i=0; i<m_Geometries.getNumberOfRenderables(); i++)
	{
	  Geometry *geo = static_cast<GeometryRenderable*>(m_Geometries.get(i))->getGeometry();
	  /*
	  if(geo->m_GeoFrame)
	    {
	      QString typestring("_tri");
	      switch(geo->m_GeoFrame->mesh_type)
		{
		case 0: typestring = "_tri"; break;
		case 1: typestring = "_tet"; break;
		case 2: typestring = "_quad"; break;
		case 3: typestring = "_hex"; break;
		case 4: typestring = "_nurbs"; break;
		}
	      
	      QString extension = QFileInfo(filename).extension();
	      QString dirpath = QFileInfo(filename).dirPath(true);
	      QString newFileName = QString("%1/%2_%3%4.%5")
		.arg(dirpath)
		.arg(QFileInfo(filename).baseName(true))
		.arg(i)
		.arg(typestring)
		.arg(extension);

	      qDebug("Writing geometry using geoframe::write_raw()");
	      geo->m_GeoFrame->write_raw(newFileName.ascii());
	    }
	  */
	  if(!geo->m_SceneGeometry.geometry.empty())
	    {
	      cvcraw_geometry::write(geo->m_SceneGeometry.geometry,
				     QString("%1%2.%3")
				       .arg(filename.remove(extension))
				       .arg(i)
				       .arg(extension)
				       .ascii());
	    }
	  else
	    {
	      if(!loader.saveFile(std::string(QString("%1%2.%3").arg(filename.remove(extension)).arg(i).arg(extension).ascii()),
				  std::string(filter.ascii()),geo))
		{
		  QMessageBox::information(this, "Error Saving File", "There was an error saving the file");
		  return;
		}
	    }
	}
    }

  /*
  for(unsigned int i = 0, Geometry *geo = static_cast<GeometryRenderable*>(m_Geometries.get(0))->getGeometry();
      geo != NULL;
      geo = static_cast<GeometryRenderable*>(m_Geometries.get(++i))->getGeometry())
    {
      
    }
  */
}

void NewVolumeMainWindow::pocketTunnelSlot()
{
#ifdef USING_POCKET_TUNNEL
  if(m_Geometries.getNumberOfRenderables() == 0)
    {
      QMessageBox::information(this, "Pocket Tunnel", "Load a geometry file first.");
      return;
    }

  if(m_PocketTunnelThread && m_PocketTunnelThread->running())
    QMessageBox::critical(this,"Error","Pocket Tunnel thread already running!",QMessageBox::Ok,Qt::NoButton,Qt::NoButton);
  else
    {
      if(m_PocketTunnelThread) delete m_PocketTunnelThread;
      m_PocketTunnelThread = 
	new PocketTunnelThread(static_cast<GeometryRenderable*>
			        (m_Geometries.getIth(m_Geometries.getNumberOfRenderables()-1))->getGeometry(),
			       this);
      m_PocketTunnelThread->start();
    }
#else
  QMessageBox::information(this, tr("Pocket Tunnel"),
			   tr("Pocket Tunnel disabled"), QMessageBox::Ok);
#endif
}

/*
**************
 */

//TODO: add a slot to perturb geometry. also figure out geom flow <-> smoothing dependency, and expose both operations if needed
void NewVolumeMainWindow::smoothGeometrySlot()
{
  if(m_Geometries.getNumberOfRenderables() == 0)
    {
      QMessageBox::information(this, "Smoothing", "Load geometry first.");
      return;
    }

  SmoothingDialog dialog;
  if(dialog.exec() != QDialog::Accepted) return;

  Geometry *geo;
  for(unsigned int i=0; i<m_Geometries.getNumberOfRenderables(); i++)
    {
      geo = static_cast<GeometryRenderable*>(m_Geometries.getIth(i))->getGeometry();
      Smoothing::smoothGeometry(geo,
				dialog.m_Delta->text().toDouble(),
				dialog.m_FixBoundary->isChecked());
      geo->CalculateTriSmoothNormals(); //recalculate normals
    }

  m_ZoomedIn->updateGL();
  m_ZoomedOut->updateGL();
  
  QMessageBox::information(this, tr("Smoothing"),
			   tr("Finished smoothing geometry."), QMessageBox::Ok);
}

void NewVolumeMainWindow::anisotropicDiffusionSlot()
{
  if(m_SourceManager.hasSource())
    {
      if(!m_RGBARendering)
	{
	  AnisotropicDiffusionDialog dialog(this);
	  
	  dialog.m_Iterations->setValidator(new QIntValidator(this));
	  
	  if(dialog.exec() == QDialog::Accepted)
	    {
	      /*
		Check if we are doing in place filtering of the actual volume data file
		instead of simply the current subvolume buffer
		TODO: need to implement out-of-core filtering using VolMagick
	      */
	      if(!dialog.m_Preview->isChecked())
		{
		  if(QMessageBox::warning(this,
					  "Anisotropic Diffusion",
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
		  vol.anisotropicDiffusion(dialog.m_Iterations->text().toInt());
		  VolMagick::writeVolumeFile(vol,
					     m_VolumeFileInfo.filename(),
					     getVarNum(),
					     getTimeStep());
		  QString tmpp(m_VolumeFileInfo.filename().c_str());
		  openFile(tmpp);
		  return;
		}
	      
	      VolumeBuffer *densityBuffer;
	      unsigned int xdim, ydim, zdim, i, len;
	      unsigned char *volume;
	      
	      densityBuffer = m_ZoomedInRenderable.getVolumeBufferManager()->getVolumeBuffer(getVarNum());
	      volume = (unsigned char *)densityBuffer->getBuffer();
	      xdim = densityBuffer->getWidth();
	      ydim = densityBuffer->getHeight();
	      zdim = densityBuffer->getDepth();
	      len = xdim*ydim*zdim;

	      VolMagickOpStatus opstatus;
	      VolMagick::Voxels vox(VolMagick::Dimension(xdim,ydim,zdim),VolMagick::UChar);
	      vox.messenger(&opstatus); //so we get a nice progress dialog during anisotropic diffusion, which will take a while
	      memcpy(*vox,volume,len*sizeof(unsigned char));

	      try
		{
		  vox.anisotropicDiffusion(dialog.m_Iterations->text().toInt());
		}
	      catch(VolMagickOpStatus::OperationCancelled &e)
		{
		  qDebug("Anisotropic diffusion operation cancelled.");
		  return;
		}

	      memcpy(volume,*vox,len*sizeof(unsigned char));

	      m_ZoomedIn->makeCurrent();
	      updateRoverRenderable(&m_ZoomedInRenderable, &m_ZoomedInExtents);
printf("NewVolumeMainWindow::anisotropicDiffusionSlot()\n");
	      m_ZoomedIn->updateGL();

	      m_SubVolumeIsFiltered = true;
	    }
	}
      else
	QMessageBox::warning(this, tr("Anisotropic Diffusion"),
			     tr("This feature is not available for RGBA volumes."), 0,0);
    }
  else
    QMessageBox::warning(this, tr("Anisotropic Diffusion"),
			 tr("A volume must be loaded for this feature to work."), 0,0);
}




void NewVolumeMainWindow::sliceRenderingSlot()
{
#ifdef VOLUMEGRIDROVER
  if(m_SourceManager.hasSource())
    {
      unsigned int v;
      SliceRenderingDialog dialog(this);
      
      dialog.m_SliceRenderingEnabled->setChecked(m_ThumbnailRenderable.sliceRendering());

      dialog.m_Variable->clear();
      for (v=0; v<m_SourceManager.getSource()->getNumVars(); v++) {
	dialog.m_Variable->insertItem(m_SourceManager.getSource()->getVariableName(v));
      }
      dialog.m_Variable->setCurrentItem(m_ThumbnailRenderable.getSliceRenderable()->currentVariable());

      dialog.m_Timestep->setMaxValue(m_SourceManager.getSource()->getNumTimeSteps()-1);
      dialog.m_Timestep->setValue(m_ThumbnailRenderable.getSliceRenderable()->currentTimestep());

      dialog.m_SliceToRenderXY->setChecked(m_ThumbnailRenderable.getSliceRenderable()->drawSlice(SliceRenderable::XY));
      dialog.m_SliceToRenderXZ->setChecked(m_ThumbnailRenderable.getSliceRenderable()->drawSlice(SliceRenderable::XZ));
      dialog.m_SliceToRenderZY->setChecked(m_ThumbnailRenderable.getSliceRenderable()->drawSlice(SliceRenderable::ZY));
      
      dialog.m_GrayscaleXY->setChecked(m_ThumbnailRenderable.getSliceRenderable()->isGrayscale(SliceRenderable::XY));
      dialog.m_GrayscaleXZ->setChecked(m_ThumbnailRenderable.getSliceRenderable()->isGrayscale(SliceRenderable::XZ));
      dialog.m_GrayscaleZY->setChecked(m_ThumbnailRenderable.getSliceRenderable()->isGrayscale(SliceRenderable::ZY));

      dialog.m_RenderAdjacentSliceXY->
	setChecked(m_ThumbnailRenderable.getSliceRenderable()->drawSecondarySlice(SliceRenderable::XY));
      dialog.m_RenderAdjacentSliceXZ->
	setChecked(m_ThumbnailRenderable.getSliceRenderable()->drawSecondarySlice(SliceRenderable::XZ));
      dialog.m_RenderAdjacentSliceZY->
	setChecked(m_ThumbnailRenderable.getSliceRenderable()->drawSecondarySlice(SliceRenderable::ZY));
       
      dialog.m_RenderAdjacentSliceOffsetXY->setValidator(new QIntValidator(this));
      dialog.m_RenderAdjacentSliceOffsetXZ->setValidator(new QIntValidator(this));
      dialog.m_RenderAdjacentSliceOffsetZY->setValidator(new QIntValidator(this));

      dialog.m_RenderAdjacentSliceOffsetXY->setText(QString("%1").
						    arg(m_ThumbnailRenderable.
							getSliceRenderable()->
							secondarySliceOffset(SliceRenderable::XY)));
      dialog.m_RenderAdjacentSliceOffsetXZ->setText(QString("%1").
						    arg(m_ThumbnailRenderable.
							getSliceRenderable()->
							secondarySliceOffset(SliceRenderable::XZ)));
      dialog.m_RenderAdjacentSliceOffsetZY->setText(QString("%1").
						    arg(m_ThumbnailRenderable.
							getSliceRenderable()->
							secondarySliceOffset(SliceRenderable::ZY)));

      dialog.m_Draw2DContoursXY->setChecked(m_ThumbnailRenderable.getSliceRenderable()->draw2DContours(SliceRenderable::XY));
      dialog.m_Draw2DContoursXZ->setChecked(m_ThumbnailRenderable.getSliceRenderable()->draw2DContours(SliceRenderable::XZ));
      dialog.m_Draw2DContoursZY->setChecked(m_ThumbnailRenderable.getSliceRenderable()->draw2DContours(SliceRenderable::ZY));

      if(dialog.exec() == QDialog::Accepted)
	{
	  m_ThumbnailRenderable.setSliceRendering(dialog.m_SliceRenderingEnabled->isChecked());
	  m_ThumbnailRenderable.getSliceRenderable()->setCurrentVariable(dialog.m_Variable->currentItem());
	  m_ThumbnailRenderable.getSliceRenderable()->setCurrentTimestep(dialog.m_Timestep->value());
	  m_ThumbnailRenderable.getSliceRenderable()->setDrawSlice(SliceRenderable::XY, dialog.m_SliceToRenderXY->isChecked());
	  m_ThumbnailRenderable.getSliceRenderable()->setDrawSlice(SliceRenderable::XZ, dialog.m_SliceToRenderXZ->isChecked());
	  m_ThumbnailRenderable.getSliceRenderable()->setDrawSlice(SliceRenderable::ZY, dialog.m_SliceToRenderZY->isChecked());
	  m_ThumbnailRenderable.getSliceRenderable()->setGrayscale(SliceRenderable::XY, dialog.m_GrayscaleXY->isChecked());
	  m_ThumbnailRenderable.getSliceRenderable()->setGrayscale(SliceRenderable::XZ, dialog.m_GrayscaleXZ->isChecked());
	  m_ThumbnailRenderable.getSliceRenderable()->setGrayscale(SliceRenderable::ZY, dialog.m_GrayscaleZY->isChecked());
	  m_ThumbnailRenderable.getSliceRenderable()->
	    setDrawSecondarySlice(SliceRenderable::XY, dialog.m_RenderAdjacentSliceXY->isChecked());
	  m_ThumbnailRenderable.getSliceRenderable()->
	    setDrawSecondarySlice(SliceRenderable::XZ, dialog.m_RenderAdjacentSliceXZ->isChecked());
	  m_ThumbnailRenderable.getSliceRenderable()->
	    setDrawSecondarySlice(SliceRenderable::ZY, dialog.m_RenderAdjacentSliceZY->isChecked());
	  m_ThumbnailRenderable.getSliceRenderable()->
	    setSecondarySliceOffset(SliceRenderable::XY, dialog.m_RenderAdjacentSliceOffsetXY->text().toInt());
	  m_ThumbnailRenderable.getSliceRenderable()->
	    setSecondarySliceOffset(SliceRenderable::XZ, dialog.m_RenderAdjacentSliceOffsetXZ->text().toInt());
	  m_ThumbnailRenderable.getSliceRenderable()->
	    setSecondarySliceOffset(SliceRenderable::ZY, dialog.m_RenderAdjacentSliceOffsetZY->text().toInt());

	  m_ThumbnailRenderable.getSliceRenderable()->
	    setDraw2DContours(SliceRenderable::XY, dialog.m_Draw2DContoursXY->isChecked());
	  m_ThumbnailRenderable.getSliceRenderable()->
	    setDraw2DContours(SliceRenderable::XZ, dialog.m_Draw2DContoursXZ->isChecked());
	  m_ThumbnailRenderable.getSliceRenderable()->
	    setDraw2DContours(SliceRenderable::ZY, dialog.m_Draw2DContoursZY->isChecked());

	  m_ZoomedInRenderable.setSliceRendering(dialog.m_SliceRenderingEnabled->isChecked());
	  m_ZoomedInRenderable.getSliceRenderable()->setCurrentVariable(dialog.m_Variable->currentItem());
	  m_ZoomedInRenderable.getSliceRenderable()->setCurrentTimestep(dialog.m_Timestep->value());
	  m_ZoomedInRenderable.getSliceRenderable()->setDrawSlice(SliceRenderable::XY, dialog.m_SliceToRenderXY->isChecked());
	  m_ZoomedInRenderable.getSliceRenderable()->setDrawSlice(SliceRenderable::XZ, dialog.m_SliceToRenderXZ->isChecked());
	  m_ZoomedInRenderable.getSliceRenderable()->setDrawSlice(SliceRenderable::ZY, dialog.m_SliceToRenderZY->isChecked());
	  m_ZoomedInRenderable.getSliceRenderable()->setGrayscale(SliceRenderable::XY, dialog.m_GrayscaleXY->isChecked());
	  m_ZoomedInRenderable.getSliceRenderable()->setGrayscale(SliceRenderable::XZ, dialog.m_GrayscaleXZ->isChecked());
	  m_ZoomedInRenderable.getSliceRenderable()->setGrayscale(SliceRenderable::ZY, dialog.m_GrayscaleZY->isChecked());
	  m_ZoomedInRenderable.getSliceRenderable()->
	    setDrawSecondarySlice(SliceRenderable::XY, dialog.m_RenderAdjacentSliceXY->isChecked());
	  m_ZoomedInRenderable.getSliceRenderable()->
	    setDrawSecondarySlice(SliceRenderable::XZ, dialog.m_RenderAdjacentSliceXZ->isChecked());
	  m_ZoomedInRenderable.getSliceRenderable()->
	    setDrawSecondarySlice(SliceRenderable::ZY, dialog.m_RenderAdjacentSliceZY->isChecked());
	  m_ZoomedInRenderable.getSliceRenderable()->
	    setSecondarySliceOffset(SliceRenderable::XY, dialog.m_RenderAdjacentSliceOffsetXY->text().toInt());
	  m_ZoomedInRenderable.getSliceRenderable()->
	    setSecondarySliceOffset(SliceRenderable::XZ, dialog.m_RenderAdjacentSliceOffsetXZ->text().toInt());
	  m_ZoomedInRenderable.getSliceRenderable()->
	    setSecondarySliceOffset(SliceRenderable::ZY, dialog.m_RenderAdjacentSliceOffsetZY->text().toInt());

	  m_ZoomedInRenderable.getSliceRenderable()->
	    setDraw2DContours(SliceRenderable::XY, dialog.m_Draw2DContoursXY->isChecked());
	  m_ZoomedInRenderable.getSliceRenderable()->
	    setDraw2DContours(SliceRenderable::XZ, dialog.m_Draw2DContoursXZ->isChecked());
	  m_ZoomedInRenderable.getSliceRenderable()->
	    setDraw2DContours(SliceRenderable::ZY, dialog.m_Draw2DContoursZY->isChecked());

	  m_ZoomedIn->updateGL();
	  m_ZoomedOut->updateGL();
	}
    }
  else
    QMessageBox::warning(this, tr("Slice Rendering"),
			 tr("A volume must be loaded for this feature to work."), 0,0);

#endif
}



void NewVolumeMainWindow::boundaryPointCloudSlot()
{
#ifdef USING_TIGHT_COCONE
  
  if(!m_SourceManager.hasSource())
    {
      QMessageBox::warning(this, tr("Boundary Point Cloud"),
			   tr("A volume must be loaded for this feature to work."), 0,0);
      return;
    }

  if(m_RGBARendering)
    {
      QMessageBox::critical(this,
			    "Error",
			    "Must be in Single volume rendering mode.",
			    QMessageBox::Ok,Qt::NoButton,Qt::NoButton);
      return;
    }
  
  BoundaryPointCloudDialog dialog;
  if(dialog.exec() != QDialog::Accepted) return;

  float tlow = dialog.m_TLow->text().toFloat();
  float thigh = dialog.m_THigh->text().toFloat();

  tlow = tlow < 0.0 ? 0.0 : tlow > 255.0 ? 255.0 : tlow;
  thigh = tlow < 0.0 ? 0.0 : thigh > 255.0 ? 255.0 : thigh;

  VolMagick::Volume vol;
  if(dialog.m_Preview->isChecked()) //if preview mode, just use the in memory volume buffer
    {
      VolumeBuffer *densityBuffer = m_ZoomedInRenderable.getVolumeBufferManager()->getVolumeBuffer(getVarNum());
      vol.dimension(VolMagick::Dimension(densityBuffer->getWidth(),
					 densityBuffer->getHeight(),
					 densityBuffer->getDepth()));
      vol.boundingBox(VolMagick::BoundingBox(m_ZoomedInExtents.getXMin(),
					     m_ZoomedInExtents.getYMin(),
					     m_ZoomedInExtents.getZMin(),
					     m_ZoomedInExtents.getXMax(),
					     m_ZoomedInExtents.getYMax(),
					     m_ZoomedInExtents.getZMax()));
      vol.voxelType(VolMagick::UChar);
      memcpy(*vol,
	     densityBuffer->getBuffer(),
	     vol.XDim()*vol.YDim()*vol.ZDim()*sizeof(unsigned char));
    }
  else
    {
      VolMagick::readVolumeFile(vol,m_VolumeFileInfo.filename(),getVarNum(),getTimeStep());
    }
  boost::shared_ptr<Geometry> result = TightCocone::generateBoundaryPointCloud(vol,tlow,thigh);
  m_Geometries.add(new GeometryRenderable(new Geometry(*result.get())));
  
  m_ZoomedIn->updateGL();
  m_ZoomedOut->updateGL();

  QMessageBox::information(this,
			   "Notice",
			   "Finished Boundary Point Cloud generation.",
			   QMessageBox::Ok);

#else
  QMessageBox::information(this, tr("Boundary Point Cloud"),
			   tr("Boundary Point Cloud (part of Tight Cocone) not built into volrover"), QMessageBox::Ok);
#endif
}

void NewVolumeMainWindow::tightCoconeSlot()
{
#ifdef USING_TIGHT_COCONE
  if(m_Geometries.getNumberOfRenderables() == 0)
    {
      QMessageBox::information(this, "Tight Cocone", "Load a geometry file first.");
      return;
    }

  TightCoconeDialog dialog;
  TightCocone::Parameters params; //for filling out defaults in the dialog
  dialog.m_EnableRobustCocone->setChecked(params.b_robust());
  dialog.m_BigBallRatio->setText(QString("%1").arg(params.bb_ratio()));
  dialog.m_ThetaIF->setText(QString("%1").arg(params.theta_if()));
  dialog.m_ThetaFF->setText(QString("%1").arg(params.theta_ff()));
  dialog.m_FlatnessRatio->setText(QString("%1").arg(params.flatness_ratio()));
  dialog.m_CoconePhi->setText(QString("%1").arg(params.cocone_phi()));
  dialog.m_FlatPhi->setText(QString("%1").arg(params.flat_phi()));
  if(dialog.exec() != QDialog::Accepted) return;

  GeometryRenderable *gr = static_cast<GeometryRenderable*>(m_Geometries.getIth(m_Geometries.getNumberOfRenderables()-1));

  boost::shared_ptr<Geometry> result =
    TightCocone::surfaceReconstruction(boost::shared_ptr<Geometry>(new Geometry(*gr->getGeometry())),
				       params.
				       b_robust(dialog.m_EnableRobustCocone->isChecked()).
				       bb_ratio(dialog.m_BigBallRatio->text().toDouble()).
				       theta_if(dialog.m_ThetaIF->text().toDouble()).
				       theta_ff(dialog.m_ThetaFF->text().toDouble()).
				       flatness_ratio(dialog.m_FlatnessRatio->text().toDouble()).
				       cocone_phi(dialog.m_CoconePhi->text().toDouble()).
				       flat_phi(dialog.m_FlatPhi->text().toDouble()));
  //m_Geometries.add(new GeometryRenderable(new Geometry(*result.get())));
  (*gr->getGeometry()) = *result.get(); //copy over input instead of add

  m_ZoomedIn->updateGL();
  m_ZoomedOut->updateGL();

  QMessageBox::information(this,
			   "Notice",
			   "Finished tight cocone.",
			   QMessageBox::Ok);

#else
  QMessageBox::information(this, tr("Tight Cocone"),
			   tr("Tight cocone not built into volrover"), QMessageBox::Ok);
#endif
}

void NewVolumeMainWindow::curationSlot()
{
#ifdef USING_CURATION
  if(m_Geometries.getNumberOfRenderables() == 0)
    {
      QMessageBox::information(this, "Curation", "Load a geometry file first.");
      return;
    }

  CurationDialog dialog;
  dialog.m_MergeRatio->setText(QString("%1").arg(Curation::DEFAULT_MERGE_RATIO));
  dialog.m_OutputSegCount->setText(QString("%1").arg(Curation::DEFAULT_OUTPUT_SEG_COUNT));
  if(dialog.exec() != QDialog::Accepted) return;

  GeometryRenderable *gr = static_cast<GeometryRenderable*>(m_Geometries.getIth(m_Geometries.getNumberOfRenderables()-1));

  std::vector<boost::shared_ptr<Geometry> > curation_result =
    Curation::curate(boost::shared_ptr<Geometry>(new Geometry(*gr->getGeometry())),
		     dialog.m_MergeRatio->text().toDouble(),
		     dialog.m_OutputSegCount->text().toInt());
  //m_Geometries.add(new GeometryRenderable(new Geometry(*curation_result[0].get())));
  (*gr->getGeometry()) = *curation_result[0].get(); //copy over input instead of add

  m_ZoomedIn->updateGL();
  m_ZoomedOut->updateGL();

  QMessageBox::information(this,
			   "Notice",
			   "Finished curation.",
			   QMessageBox::Ok);

#else
  QMessageBox::information(this, tr("Curation"),
			   tr("Curation not built into volrover"), QMessageBox::Ok);
#endif
}

void NewVolumeMainWindow::skeletonizationSlot()
{
#ifdef USING_SKELETONIZATION
  if(m_Geometries.getNumberOfRenderables() == 0)
    {
      QMessageBox::information(this, "Skeletonization", "Load a geometry file first.");
      return;
    }

  SkeletonizationDialog dialog;
  Skeletonization::Parameters params; //for filling out defaults in the dialog
  dialog.m_EnableRobustCocone->setChecked(params.b_robust());
  dialog.m_BigBallRatio->setText(QString("%1").arg(params.bb_ratio()));
  dialog.m_ThetaIF->setText(QString("%1").arg(params.theta_if()));
  dialog.m_ThetaFF->setText(QString("%1").arg(params.theta_ff()));
  dialog.m_FlatnessRatio->setText(QString("%1").arg(params.flatness_ratio()));
  dialog.m_CoconePhi->setText(QString("%1").arg(params.cocone_phi()));
  dialog.m_FlatPhi->setText(QString("%1").arg(params.flat_phi()));
  dialog.m_Threshold->setText(QString("%1").arg(params.threshold()));
  dialog.m_PlCnt->setText(QString("%1").arg(params.pl_cnt()));
  dialog.m_DiscardByThreshold->setChecked(params.discard_by_threshold());
  dialog.m_Theta->setText(QString("%1").arg(params.theta()));
  dialog.m_MedialRatio->setText(QString("%1").arg(params.medial_ratio()));

  if(dialog.exec() == QDialog::Accepted)
    {
      GeometryRenderable *gr = 
	static_cast<GeometryRenderable*>(m_Geometries.getIth(m_Geometries.getNumberOfRenderables()-1));
      
      Skeletonization::Simple_skel result = 
	Skeletonization::skeletonize(boost::shared_ptr<Geometry>(new Geometry(*gr->getGeometry())),
				     params.
				     b_robust(dialog.m_EnableRobustCocone->isChecked()).
				     bb_ratio(dialog.m_BigBallRatio->text().toDouble()).
				     theta_if(dialog.m_ThetaIF->text().toDouble()).
				     theta_ff(dialog.m_ThetaFF->text().toDouble()).
				     flatness_ratio(dialog.m_FlatnessRatio->text().toDouble()).
				     cocone_phi(dialog.m_CoconePhi->text().toDouble()).
				     flat_phi(dialog.m_FlatPhi->text().toDouble()).
				     threshold(dialog.m_Threshold->text().toDouble()).
				     pl_cnt(dialog.m_PlCnt->text().toDouble()).
				     discard_by_threshold(dialog.m_DiscardByThreshold->isChecked()).
				     theta(dialog.m_Theta->text().toDouble()).
				     medial_ratio(dialog.m_MedialRatio->text().toDouble()));
      
      m_ThumbnailRenderable.getSkeletonRenderable()->skel(result);
      m_ZoomedInRenderable.getSkeletonRenderable()->skel(result);
      
      m_ZoomedIn->updateGL();
      m_ZoomedOut->updateGL();
      
      QMessageBox::information(this,
			       "Notice",
			       "Finished skeletonization.",
			       QMessageBox::Ok);
    }
#else
  QMessageBox::information(this, tr("Skeletonization"),
			   tr("Skeletonization not built into volrover"), QMessageBox::Ok);
#endif
}



void NewVolumeMainWindow::secondaryStructureElucidationSlot( )
{
	Geometry* inputGeometry = 0;
#ifdef USING_SECONDARYSTRUCTURES
	if(m_Geometries.getNumberOfRenderables()!= 0)
	{ 
	   	if(m_Geometries.getNumberOfRenderables() >1)
        {
			switch(QMessageBox::warning(this, "Warning", "You have loaded several geometries. I only work for the first geometry. You are strongly recommended to delete the unnecessary ones.", tr("&Continue"), tr("&Cancel"), 0, 0, 1))
			{
			case 0: 
			     break;
			case 1:
			 	return;
		    }
	      }
		GeometryRenderable* gr = static_cast <GeometryRenderable*> (m_Geometries.getIth(0)); //m_Geometries.getNumberOfRenderables()-1));
		inputGeometry = gr->getGeometry();
	}
	else if(m_SourceManager.hasSource()){
		if(!m_RGBARendering)
		{
			IsocontourMap isocontours = m_ColorTable->getIsocontourMap();
			if(isocontours.GetSize() ==1)
				inputGeometry = m_ThumbnailRenderable.getMultiContour()->getGeometry();
			else if (isocontours.GetSize() > 1)
			{
				switch(QMessageBox::warning(this, tr("Warning"), tr("You have selected multi contours. I only work for the first isocontour surface. You are strongly recommended to delete the other isocontour nodes."), tr("&Continue"), tr("&Cancel"), 0, 0, 1))
				{
				 case 0:
			
				   for(int i = 2; i<=isocontours.GetSize(); i++)
				   {
				   		m_ThumbnailRenderable.getMultiContour()->removeContour(i);
						m_ZoomedInRenderable.getMultiContour()->removeContour(i);
				   }
				   inputGeometry = m_ThumbnailRenderable.getMultiContour()->getGeometry();
				   syncIsocontourValuesWithVolumeGridRover();
				   m_ZoomedIn->updateGL();
				   m_ZoomedOut->updateGL();
				    break;
				 case 1:
				    return;
				}
			}
			else 
			{
				QMessageBox::warning(this, tr("Warning"), tr("No isosurface is selected!"));
				return;
			}

			GeometryRenderable* gr0 = new GeometryRenderable(inputGeometry);
			m_Geometries.add(gr0);
		}
	}
	else
	{
		QMessageBox::information(this, "Notice", "Scalar volume or surface not currently selected");
		return;
	}

	assert(inputGeometry!=0);


    ssData = new SecondaryStructureData(inputGeometry, &m_Geometries, m_ZoomedIn, m_ZoomedOut );

	
#endif
}

void NewVolumeMainWindow::clipGeometryToVolumeBoxSlot(bool clip)
{
  m_ThumbnailRenderable.getGeometryRenderer()->setClipGeometry(clip);
  m_ZoomedInRenderable.getGeometryRenderer()->setClipGeometry(clip);
#ifdef VOLUMEGRIDROVER
  m_ThumbnailRenderable.getSliceRenderable()->setClipGeometry(clip);
  m_ZoomedInRenderable.getSliceRenderable()->setClipGeometry(clip);
#endif
#ifdef USING_SKELETONIZATION
  m_ThumbnailRenderable.getSkeletonRenderable()->setClipGeometry(clip);
  m_ZoomedInRenderable.getSkeletonRenderable()->setClipGeometry(clip);
#endif
  m_ZoomedIn->updateGL();
  m_ZoomedOut->updateGL();
}

void NewVolumeMainWindow::saveSkeletonSlot()
{
#ifdef USING_SKELETONIZATION
  SkeletonRenderable *skel_ren = m_ThumbnailRenderable.getSkeletonRenderable();
  if(!skel_ren->skel_polys() || !skel_ren->skel_lines())
    {
      QMessageBox::warning(this, tr("Save Skeleton"),
			   tr("No skeleton!."), 0, 0);
      return;
    }

  QString filename = Q3FileDialog::getSaveFileName(QString::null, 
						  "All Files (*)", 
						  this, "Save As", "Save As File Prefix");
  if(filename.isNull()) return;

  QString polys_filename = filename + ".rawc";
  QString lines_filename = filename + ".linec";

  if(QFileInfo(polys_filename).exists() && 
     QMessageBox::No ==
     QMessageBox::question(this,
			   "File exists!",
			   QString("File %1 exists, overwrite?").arg(polys_filename),
			   QMessageBox::No,
			   QMessageBox::Yes))
    return;

  if(QFileInfo(lines_filename).exists() && 
     QMessageBox::No ==
     QMessageBox::question(this,
			   "File exists!",
			   QString("File %1 exists, overwrite?").arg(lines_filename),
			   QMessageBox::No,
			   QMessageBox::Yes))
    return;

  GeometryLoader loader;
  if(!loader.saveFile(polys_filename,
		      "Rawc files (*.rawc)",
		      const_cast<Geometry*>(skel_ren->skel_polys()->getGeometry())))
    {
      QMessageBox::information(this, "Error Saving File", "There was an error saving the file");
      return;
    }

  if(!loader.saveFile(lines_filename,
		      "Linec files (*.linec)",
		      const_cast<Geometry*>(skel_ren->skel_lines()->getGeometry())))
    {
      QMessageBox::information(this, "Error Saving File", "There was an error saving the file");
      return;
    }
#else
  QMessageBox::information(this, "Error Saving File", "Skeletonization support not built into VolRover!");
  return;
#endif
}

void NewVolumeMainWindow::clearSkeletonSlot()
{
#ifdef USING_SKELETONIZATION
  m_ThumbnailRenderable.getSkeletonRenderable()->clear();
  m_ZoomedInRenderable.getSkeletonRenderable()->clear();
  m_ZoomedIn->updateGL();
  m_ZoomedOut->updateGL();
#else
  QMessageBox::information(this, "Error Saving File", "Skeletonization support not built into VolRover!");
  return;
#endif
}

void NewVolumeMainWindow::signedDistanceFunctionSlot()
{
  if(m_Geometries.getNumberOfRenderables() == 0) 
    { 
      QMessageBox::information(this, "Signed Distance Function", "Load a geometry file first."); 
      return; 
    }

  SignedDistanceFunctionDialog dialog;
  
  //fill in the manual bounding box line edits with the geometry extents
  {
    VolMagick::BoundingBox globalbox;
    Geometry *geo;
    for(unsigned int i=0; i<m_Geometries.getNumberOfRenderables(); i++)
      {
	geo = static_cast<GeometryRenderable*>(m_Geometries.get(i))->getGeometry();
	if(geo->m_GeoFrame)
	  {
	    geo->m_GeoFrame->calculateExtents();
	    globalbox += VolMagick::BoundingBox(geo->m_GeoFrame->min_x,
						geo->m_GeoFrame->min_y,
						geo->m_GeoFrame->min_z,
						geo->m_GeoFrame->max_x,
						geo->m_GeoFrame->max_y,
						geo->m_GeoFrame->max_z);
	  }
	else
	  {
	    geo->GetReadyToDrawWire(); //calculate extents if needed
	    globalbox += VolMagick::BoundingBox(geo->m_Min[0],geo->m_Min[1],geo->m_Min[2],
						geo->m_Max[0],geo->m_Max[1],geo->m_Max[2]);
	  }
      }

    dialog.m_MinX->setText(QString("%1").arg(globalbox.minx));
    dialog.m_MinY->setText(QString("%1").arg(globalbox.miny));
    dialog.m_MinZ->setText(QString("%1").arg(globalbox.minz));
    dialog.m_MaxX->setText(QString("%1").arg(globalbox.maxx));
    dialog.m_MaxY->setText(QString("%1").arg(globalbox.maxy));
    dialog.m_MaxZ->setText(QString("%1").arg(globalbox.maxz));
  }

  if(dialog.exec() == QDialog::Accepted)
    {
      try
	{
	  VolMagick::Dimension dim;
	  VolMagick::BoundingBox bbox;

	  dim = VolMagick::Dimension(VolMagick::uint64(dialog.m_DimX->text().toInt()),
				     VolMagick::uint64(dialog.m_DimY->text().toInt()),
				     VolMagick::uint64(dialog.m_DimZ->text().toInt()));
	  if(dialog.m_UseBoundingBox->isChecked())
	    bbox = VolMagick::BoundingBox(dialog.m_MinX->text().toDouble(),
					  dialog.m_MinY->text().toDouble(),
					  dialog.m_MinZ->text().toDouble(),
					  dialog.m_MaxX->text().toDouble(),
					  dialog.m_MaxY->text().toDouble(),
					  dialog.m_MaxZ->text().toDouble());

	  GeometryRenderable *gr = 
	    static_cast<GeometryRenderable*>(m_Geometries.getIth(m_Geometries.getNumberOfRenderables()-1));

	  //actually perform the computation using the selected method
	  VolMagick::Volume vol;
	  switch(dialog.m_Method->currentItem())
	    {
	    case 0:
	      vol = multi_sdf::signedDistanceFunction(boost::shared_ptr<Geometry>(new Geometry(*gr->getGeometry())),dim,bbox);
	      vol.desc("Signed Distance Function - multi_sdf");
	      break;
	    case 1:
	      vol = SDFLibrary::signedDistanceFunction(boost::shared_ptr<Geometry>(new Geometry(*gr->getGeometry())),dim,bbox);
	      vol.desc("Signed Distance Function - SDFLibrary");
	      break;
	    }

     
	  QDir tmpdir(m_CacheDir.absPath() + "/tmp");
	  if(!tmpdir.exists()) tmpdir.mkdir(tmpdir.absPath());
	  QFileInfo tmpfile(tmpdir,"tmp.rawv");
	  qDebug("Creating volume %s",tmpfile.absFilePath().ascii());
	  QString newfilename = QDir::convertSeparators(tmpfile.absFilePath());
	  VolMagick::createVolumeFile(newfilename.toStdString(),
				      vol.boundingBox(),
				      vol.dimension(),
				      std::vector<VolMagick::VoxelType>(1, vol.voxelType()));
	  VolMagick::writeVolumeFile(vol,newfilename.toStdString());
	  openFile(newfilename);
	}
      catch(const VolMagick::Exception& e)
	{
	  QMessageBox::critical(this,"Error loading SDF volume",e.what());
	  return;
	}
    }
}

void NewVolumeMainWindow::mergeGeometrySlot()
{
  Geometry all;
  for(unsigned int i=0; i<m_Geometries.getNumberOfRenderables(); i++)
    all.merge(*static_cast<GeometryRenderable*>(m_Geometries.getIth(i))->getGeometry());
  m_Geometries.clear();
  m_Geometries.add(new GeometryRenderable(new Geometry(all)));
  m_ZoomedIn->updateGL();
  m_ZoomedOut->updateGL();
  QMessageBox::information(this, "Merge Geometry", "Merge geometry complete."); 
}

void NewVolumeMainWindow::convertIsosurfaceToGeometrySlot()
{
  ConvertIsosurfaceToGeometryDialogBase dialog;

  if(dialog.exec() == QDialog::Accepted)
    {
      Geometry *geo;
      switch(dialog.m_IsosurfaceConversionOptionsGroup->selectedId())
	{
	case 0:
	  geo = m_ZoomedInRenderable.getMultiContour()->getGeometry();
	  break;
	case 1:
	  geo = m_ThumbnailRenderable.getMultiContour()->getGeometry();
	  break;
	}

      if(!geo)
	{
	  QMessageBox::warning(this, tr("Convert Isosurface"),
			       tr("There are no isosurfaces to convert!"), 0,0);
	  return;
	}

      GeometryRenderable *gr = new GeometryRenderable(geo);
      gr->setWireframeMode(WireframeRenderToggleAction->isOn());
      gr->setSurfWithWire(RenderSurfaceWithWireframeAction->isOn());
      m_Geometries.add(gr);
      m_ZoomedIn->updateGL();
      m_ZoomedOut->updateGL();
    }
}


void NewVolumeMainWindow::highLevelSetReconSlot()
{
  qDebug("highLevelSetReconSlot()");

#ifdef USING_HLEVELSET
  unsigned int dim[3];
  //  extern float edgelength,end;

  if(m_Geometries.getNumberOfRenderables() == 0) 
    { 
      QMessageBox::information(this, "High Level Set Reconstruction", "Load a geometry file first."); 
      return; 
    }

  HighlevelsetReconDialog dialog(this);
 
  dialog.m_dimEdit->setValidator(new QDoubleValidator(this));
  dialog.m_edgelengthEdit->setValidator(new QDoubleValidator(this));
  dialog.m_endEdit->setValidator(new QDoubleValidator(this));
  dialog.m_MaxdimEdit->setValidator(new QDoubleValidator(this));

  if(dialog.exec() == QDialog::Accepted)
    {
      dim[0] = dim[1] = dim[2] = atoi(dialog.m_dimEdit->text().ascii());
      edgelength = atof(dialog.m_edgelengthEdit->text().ascii());
      end = atoi(dialog.m_endEdit->text().ascii());
      Max_dim = atoi(dialog.m_MaxdimEdit->text().ascii());
      //       printf("OK!");       
      //  dim[0] = dim[1] = dim[2] = 256;//128;
      HLevelSetNS::HLevelSet* hLevelSet = new HLevelSetNS::HLevelSet();

      GeometryRenderable *gr = 
	static_cast<GeometryRenderable*>(m_Geometries.getIth(m_Geometries.getNumberOfRenderables()-1));

      boost::shared_ptr<Geometry> geometry(new Geometry(*gr->getGeometry()));
      std::vector<float> vertex_Positions(geometry->m_NumPoints*3);
      for(unsigned int i = 0; i < geometry->m_NumPoints; i++)
	{
	  vertex_Positions[i*3+0] = geometry->m_Points[i*3+0];
	  vertex_Positions[i*3+1] = geometry->m_Points[i*3+1];
	  vertex_Positions[i*3+2] = geometry->m_Points[i*3+2];
	}
  
      //  printf("vertex_=%f %f %f ",vertex_Positions[38581*3+0],vertex_Positions[38581*3+0],vertex_Positions[38581*3+2]);getchar();
      clock_t t;
      double time;
      t=clock();
      time=(double)t/(double)CLOCKS_PER_SEC;

      boost::tuple<bool,VolMagick::Volume> result = 
	hLevelSet->getHigherOrderLevelSetSurface_Xu_Li(vertex_Positions,dim);
      delete hLevelSet;

      clock_t t_end;
      double time_end;
      t_end=clock();
      time_end=(double)t_end/(double)CLOCKS_PER_SEC;
      time=time_end-time;
      printf("\n Highlevelset Time = %f \n",time);

      //  printf("OK2"); 
      try
	{
	  VolMagick::Volume result_vol(result.get<1>());
	  QDir tmpdir(m_CacheDir.absPath() + "/tmp");
	  if(!tmpdir.exists()) tmpdir.mkdir(tmpdir.absPath());
	  QFileInfo tmpfile(tmpdir,"tmp.rawiv");
	  qDebug("Creating volume %s",tmpfile.absFilePath().ascii());
	  QString newfilename = QDir::convertSeparators(tmpfile.absFilePath());
	  VolMagick::createVolumeFile(newfilename.toStdString(),
				      result_vol.boundingBox(),
				      result_vol.dimension(),
				      std::vector<VolMagick::VoxelType>(1, result_vol.voxelType()));
	  VolMagick::writeVolumeFile(result_vol,newfilename.toStd.String());
	  openFile(newfilename);
	}
      catch(const VolMagick::Exception& e)
	{
	  QMessageBox::critical(this,"Error loading HLS volume",e.what());
	  return;
	}
    }
#endif
}


void NewVolumeMainWindow::highLevelSetSlot()
{
#ifdef USING_HLEVELSET

  if(m_Geometries.getNumberOfRenderables() == 0) 
    { 
      QMessageBox::information(this, "High Level Set", "Load a geometry file first."); 
      return; 
    }

 unsigned int dim[3];
 dim[0] = dim[1] = dim[2] = 400;//256;//128;

 HLevelSetNS::HLevelSet* hLevelSet = new HLevelSetNS::HLevelSet();

 GeometryRenderable *gr = 
    static_cast<GeometryRenderable*>(m_Geometries.getIth(m_Geometries.getNumberOfRenderables()-1));

  boost::shared_ptr<Geometry> geometry(new Geometry(*gr->getGeometry()));
  std::vector<float> vertex_Positions(geometry->m_NumPoints*3);
  for(unsigned int i = 0; i < geometry->m_NumPoints; i++)
    {
      vertex_Positions[i*3+0] = geometry->m_Points[i*3+0];
      vertex_Positions[i*3+1] = geometry->m_Points[i*3+1];
      vertex_Positions[i*3+2] = geometry->m_Points[i*3+2];
    }

  std::vector<float> radii(geometry->m_NumPoints);
  for(unsigned int i = 0; i < geometry->m_NumPoints; i++)
    {
      radii[i] = geometry->m_PointScalars[i];
    }

  //  printf("vertex_=%f %f %f ",vertex_Positions[38581*3+0],vertex_Positions[38581*3+0],vertex_Positions[38581*3+2]);getchar();
  clock_t t;
  double time;
  t=clock();
  time=(double)t/(double)CLOCKS_PER_SEC;
  
  boost::tuple<bool,VolMagick::Volume> result =
    hLevelSet->getHigherOrderLevelSetSurface(vertex_Positions, radii,dim );
  delete hLevelSet;
  
  clock_t t_end;
  double time_end;
  t_end=clock();
  time_end=(double)t_end/(double)CLOCKS_PER_SEC;
  time=time_end-time;
  printf("\n Highlevelset Time = %f \n",time);
  
  try
    {
      VolMagick::Volume result_vol(result.get<1>());
      QDir tmpdir(m_CacheDir.absPath() + "/tmp");
      if(!tmpdir.exists()) tmpdir.mkdir(tmpdir.absPath());
      QFileInfo tmpfile(tmpdir,"tmp.rawiv");
      qDebug("Creating volume %s",tmpfile.absFilePath().ascii());
      QString newfilename = QDir::convertSeparators(tmpfile.absFilePath());
      VolMagick::createVolumeFile(newfilename.toStdString(),
				  result_vol.boundingBox(),
				  result_vol.dimension(),
				  std::vector<VolMagick::VoxelType>(1, result_vol.voxelType()));
      VolMagick::writeVolumeFile(result_vol,newfilename.toStdString());
      openFile(newfilename);
    }
  catch(const VolMagick::Exception& e)
    {
      QMessageBox::critical(this,"Error loading HLS volume",e.what());
      return;
    }
#endif
}

void NewVolumeMainWindow::LBIEMeshingSlot()
{
  if(!m_SourceManager.hasSource())
    {
      QMessageBox::warning(this, tr("LBIE Meshing"),
			   tr("A volume must be loaded for this feature to work."), 0,0);
      return;
    }
  
  if(m_RGBARendering)
    {
      QMessageBox::critical(this,
			    "LBIE Meshing",
			    "Must be in Single volume rendering mode.",
			    QMessageBox::Ok,Qt::NoButton,Qt::NoButton);
      return;
    }

  LBIEMeshingDialog dialog;
  if(dialog.exec() != QDialog::Accepted) return;
  
  //LBIE isovalues appear to be real voxel values (i.e. not mapped to [0.0-1.0] or [0.0-255.0])
  float outer_isoval, inner_isoval;
  switch(dialog.m_WhichIsovaluesGroup->selectedId())
    {
    case 0:
      {
	IsocontourMap isocontours = m_ColorTable->getIsocontourMap();
	if(isocontours.GetSize() < 1)
	  {
	    QMessageBox::critical(this,
				  "LBIE Meshing",
				  "No isocontour nodes specified in color table!",
				  QMessageBox::Ok,Qt::NoButton,Qt::NoButton);
	    return;
	  }
	std::vector<double> values(isocontours.GetSize());
	for(int i = 0; i < isocontours.GetSize(); i++)
	  values.push_back(m_VolumeFileInfo.min() + 
			   isocontours.GetPositionOfIthNode(i)*(m_VolumeFileInfo.max()-m_VolumeFileInfo.min()));
	values.erase(values.begin()); //always a 0.0 value for some reason
	if(values.empty())
	  {
	    QMessageBox::critical(this,
				  "LBIE Meshing",
				  "Please insert an isocontour bar into the color table below.",
				  QMessageBox::Ok,Qt::NoButton,Qt::NoButton);
	    return;
	  }
	
	outer_isoval = *std::min_element(values.begin(),values.end());
	inner_isoval = *std::max_element(values.begin(),values.end());
      }
      break;
    case 1:
      {
	outer_isoval = dialog.m_OuterIsoValue->text().toDouble();
	inner_isoval = dialog.m_InnerIsoValue->text().toDouble();
      }
      break;
    default: break;
    }

  //They also appear to always be the opposite sign that they should be...
  //  outer_isoval *= -1;
  //  inner_isoval *= -1;
  //NOTE: library interface has been changed, signs aren't opposite anymore

  qDebug("outer_isoval == %f",outer_isoval);
  qDebug("inner_isoval == %f",inner_isoval);

  VolMagick::Volume vol;
  if(dialog.m_Preview->isChecked()) //if preview mode, just use the in memory volume buffer
    {
      VolumeBuffer *densityBuffer = m_ZoomedInRenderable.getVolumeBufferManager()->getVolumeBuffer(getVarNum());
      vol.dimension(VolMagick::Dimension(densityBuffer->getWidth(),
					 densityBuffer->getHeight(),
					 densityBuffer->getDepth()));
      vol.boundingBox(VolMagick::BoundingBox(m_ZoomedInExtents.getXMin(),
					     m_ZoomedInExtents.getYMin(),
					     m_ZoomedInExtents.getZMin(),
					     m_ZoomedInExtents.getXMax(),
					     m_ZoomedInExtents.getYMax(),
					     m_ZoomedInExtents.getZMax()));
      vol.voxelType(VolMagick::UChar);
      memcpy(*vol,
	     densityBuffer->getBuffer(),
	     vol.XDim()*vol.YDim()*vol.ZDim()*sizeof(unsigned char));
    }
  else
    {
      VolMagick::readVolumeFile(vol,m_VolumeFileInfo.filename(),getVarNum(),getTimeStep());
    }

#if 0
  //LBIE requires volumes with dimension (2^n+1)^3
  unsigned int dim[3] = { vol.XDim(), vol.YDim(), vol.ZDim() };
  unsigned int maxdim = *std::max_element(dim,dim+3);
  qDebug("maxdim = %d",maxdim);
  if((dim[0] != dim[1]) ||
     (dim[0] != dim[2]) ||
     upToPowerOfTwo(maxdim-1) != (maxdim-1))
    {
      if(QMessageBox::warning(this,
			      "LBIE Meshing",
			      QString("Volume must be resized to dimension (2^n+1)^3 (%1). Proceed?")
			      .arg(upToPowerOfTwo(maxdim)+1),
			      QMessageBox::Cancel | QMessageBox::Default,
			      QMessageBox::Ok) == QMessageBox::Cancel) return;
      vol.resize(VolMagick::Dimension(upToPowerOfTwo(maxdim)+1,
				      upToPowerOfTwo(maxdim)+1,
				      upToPowerOfTwo(maxdim)+1));
    }
#endif

#if 0
  boost::shared_ptr<Geometry> result = LBIE::mesh(vol,
						  outer_isoval,
						  inner_isoval,
						  dialog.m_ErrorTolerance->text().toDouble());
#endif

  LBIE::Mesher mesher(outer_isoval,
		      inner_isoval,
		      dialog.m_ErrorTolerance->text().toDouble(),
		      dialog.m_InnerErrorTolerance->text().toDouble(),
		      LBIE::Mesher::MeshType(dialog.m_MeshType->currentItem()),
		      LBIE::Mesher::ImproveMethod(dialog.m_ImproveMethod->currentItem()),
		      LBIE::Mesher::NormalType(dialog.m_NormalType->currentItem()),
		      LBIE::Mesher::ExtractionMethod(dialog.m_MeshExtractionMethod->currentItem()),
		      dialog.m_DualContouring->isChecked());
  mesher.extractMesh(vol);
  mesher.qualityImprove(dialog.m_Iterations->text().toInt());
  Geometry *result = new Geometry();
  //result->m_GeoFrame.reset(new LBIE::geoframe(mesher.mesh()));
  if(mesher.mesh().mesh_type == LBIE::geoframe::SINGLE)
    LBIE::copyGeoframeToGeometry(mesher.mesh(), *result);
  else
    result->m_GeoFrame.reset(new LBIE::geoframe(mesher.mesh())); 
  //mesher.mesh().write_raw("/tmp/test.raw");
  GeometryRenderable *gr = new GeometryRenderable(result);
  gr->setWireframeMode(WireframeRenderToggleAction->isOn());
  gr->setSurfWithWire(RenderSurfaceWithWireframeAction->isOn());
  m_Geometries.add(gr);

  m_ZoomedIn->updateGL();
  m_ZoomedOut->updateGL();

  QMessageBox::information(this,
			   "Notice",
			   "Finished LBIE Meshing",
			   QMessageBox::Ok);
}

void NewVolumeMainWindow::LBIEQualityImprovementSlot()
{
  if(m_Geometries.getNumberOfRenderables() == 0)
    {
      QMessageBox::information(this, "LBIE Quality Improvement", "Load a geometry file first.");
      return;
    }

  LBIEQualityImprovementDialog dialog;
  if(dialog.exec() != QDialog::Accepted) return;

  for(unsigned int i=0; i<m_Geometries.getNumberOfRenderables(); i++)
    {
      GeometryRenderable *gr = static_cast<GeometryRenderable*>(m_Geometries.getIth(i));
      Geometry input(*gr->getGeometry());
      if(!input.m_GeoFrame)
	{
	  input.m_GeoFrame.reset(new LBIE::geoframe());
	  LBIE::copyGeometryToGeoframe(input,*input.m_GeoFrame.get());
	  input.ClearGeometry();
	  //input.m_GeoFrame->write_raw("/tmp/test.raw",0);
	}
      LBIE::Mesher mesher;
      mesher.mesh(*input.m_GeoFrame.get());
      mesher.improveMethod(LBIE::Mesher::ImproveMethod(dialog.m_ImproveMethod->currentItem()));
      *input.m_GeoFrame.get() = mesher.qualityImprove(dialog.m_Iterations->text().toInt());
      if(input.m_GeoFrame->mesh_type == 0) //triangle surface rendering of geoframe seems broken
	{
	  LBIE::copyGeoframeToGeometry(*input.m_GeoFrame.get(),
				       input);
	  input.m_GeoFrame.reset();
	  
	  //the number of vertices stays the same, so lets copy the colors from the original
	  //input.m_PointColors = gr->getGeometry()->m_PointColors;
	}
      *gr->getGeometry() = input;
    }

  m_ZoomedIn->updateGL();
  m_ZoomedOut->updateGL();

  QMessageBox::information(this,
			   "Notice",
			   "Finished LBIE Quality Improvement.",
			   QMessageBox::Ok);
}

#warning TODO: make openImageFileSlot() out of core
void NewVolumeMainWindow::openImageFileSlot()
{
  using boost::any_cast;
  using boost::any;

  /*
    The effective luminance of a pixel is calculated with the following formula:
    Y=0.3RED+0.59GREEN+0.11Blue
  */

  QStringList files = Q3FileDialog::getOpenFileNames("Images (*.png *.jpg *.jpeg *.bmp *.xbm *.xpm *.pnm *.mng *.gif);;"
						    "2D MRC (*.mrc *.map)",
						    QString::null,
						    this,
						    "Open Images",
						    "Select one or more images to open");

  QStringList list = files;
  QStringList::Iterator it = list.begin();
  std::vector<any> images;
  while( it != list.end() )
    {
      //use volmagick to load mrc, and QImage to load anything else
      if((*it).endsWith(".mrc") || (*it).endsWith(".map"))
	{
	  try
	    {
	      VolMagick::Volume mrc_vol;
	      VolMagick::readVolumeFile(mrc_vol,(*it).ascii());
	      mrc_vol.map(0.0,255.0);
	      mrc_vol.voxelType(VolMagick::UChar);
	      if(mrc_vol.ZDim() > 1)
		qDebug("Warning: MRC file is 3D map.  Consider loading the file via 'Open Volume File' menu option");
	      images.push_back(mrc_vol);
	    }
	  catch(VolMagick::Exception &e)
	    {
	      QMessageBox::critical(this,"Error opening the file",e.what());
	    }
	}
      else
	{
	  QImage img(*it);
	  if(!img.isNull())
	    images.push_back(img);
	  else
	    QMessageBox::critical(this,
				  "Open Image Files",
				  QString("Could not open file: %1").arg(*it),
				  QMessageBox::Ok,Qt::NoButton,Qt::NoButton);
	}
      ++it;
    }

  if(images.empty()) return;
  while(images.size() < 4) //use at least 4 slices so the cache works correctly...
    images.push_back(images.back());

  //find the minimum dimensions
  int min_width, min_height;
  if(images[0].type() == typeid(QImage))
    {
      QImage img = any_cast<QImage>(images[0]);
      min_width = img.width();
      min_height = img.height();
    }
  else
    {
      VolMagick::Volume img = any_cast<VolMagick::Volume>(images[0]);
      min_width = img.XDim();
      min_height = img.YDim();
    }

  for(std::vector<any>::iterator i = images.begin();
      i != images.end();
      i++)
    {
      if(i->type() == typeid(QImage))
	{
	  QImage img = any_cast<QImage>(*i);
	  if(min_width > img.width()) min_width = img.width();
	  if(min_height > img.height()) min_height = img.height();
	}
      else
	{
	  VolMagick::Volume img = any_cast<VolMagick::Volume>(*i);
	  if(min_width > img.XDim()) min_width = img.XDim();
	  if(min_height > img.YDim()) min_height = img.YDim();
	}
    }
  
  VolMagick::Volume vol(VolMagick::Dimension(min_width,min_height,images.size()),
			VolMagick::UChar,
			VolMagick::BoundingBox(0.0,0.0,0.0,
					       min_width-1.0,min_height-1.0,images.size()-1.0));

  //now convert all images to grayscale, then stack them as a volume
  for(std::vector<any>::iterator k = images.begin();
      k != images.end();
      k++)
    {
      if(k->type() == typeid(QImage))
	{
	  QImage img = any_cast<QImage>(*k);
	  for(int i = 0; i < min_width; i++)
	    for(int j = 0; j < min_height; j++)
	      vol(i,j,std::distance(images.begin(),k), qGray(img.pixel(i,j)));
	}
      else
	{
	  VolMagick::Volume img = any_cast<VolMagick::Volume>(*k);
	  for(int i = 0; i < min_width; i++)
	    for(int j = 0; j < min_height; j++)
	      vol(i,j,std::distance(images.begin(),k), img(i,j,0));
	}
    }

  //finally write out the volume to the temp location and load it
  try
    {
      QDir tmpdir(m_CacheDir.absPath() + "/tmp");
      if(!tmpdir.exists()) tmpdir.mkdir(tmpdir.absPath());
      QFileInfo tmpfile(tmpdir,"tmp.rawiv");
      qDebug("Creating volume %s",tmpfile.absFilePath().ascii());
      QString newfilename = QDir::convertSeparators(tmpfile.absFilePath());
      VolMagick::createVolumeFile(newfilename.ascii(),
				  vol.boundingBox(),
				  vol.dimension(),
				  std::vector<VolMagick::VoxelType>(1, vol.voxelType()));
      VolMagick::writeVolumeFile(vol,newfilename.toStdString());
      openFile(newfilename);
    }
  catch(const VolMagick::Exception& e)
    {
      QMessageBox::critical(this,"Error opening the file",e.what());
      return;
    }
}

void NewVolumeMainWindow::projectGeometrySlot()
{
  using namespace std;

  ProjectGeometryDialog dialog;
  if(dialog.exec() == QDialog::Accepted)
    {
      try
	{
	  typedef cvcraw_geometry::geometry_t::point_t point_t;
	  typedef cvcraw_geometry::geometry_t::triangle_t triangle_t;
	  cvcraw_geometry::geometry_t reference;
	  cvcraw_geometry::read(reference,dialog.m_FileName->text().ascii());

	  Geometry *geo;
	  for(unsigned int i = 0; i < m_Geometries.getNumberOfRenderables(); i++)
	    {
	      //collect all the border vertices as those are the only ones to be projected
	      vector<point_t> border_verts;
	      vector<triangle_t> border_tris;//TODO: fill this up with geoframe stuff if needed
	      vector<unsigned int> border_vert_indices;
	      geo = static_cast<GeometryRenderable*>(m_Geometries.getIth(i))->getGeometry();
	      if(geo->m_GeoFrame) /* geoframe case */
		{
		  for(unsigned int i = 0; i < geo->m_GeoFrame->numverts; i++)
		    {
		      point_t vert = {{ geo->m_GeoFrame->verts[i][0],
					geo->m_GeoFrame->verts[i][1],
					geo->m_GeoFrame->verts[i][2] }};
		      if(geo->m_GeoFrame->bound_sign[i])
			{
			  border_verts.push_back(vert);
			  border_vert_indices.push_back(i);
			}
		    }
		}
	      else /* Geometry case */
		{
		  for(unsigned int i = 0; i < geo->m_NumPoints; i++)
		    {
		      point_t vert = {{ geo->m_Points[i*3+0],
					geo->m_Points[i*3+1],
					geo->m_Points[i*3+2] }};
		      border_verts.push_back(vert);
		    }

		  for(unsigned int i = 0; i < geo->m_NumTris; i++)
		    {
		      triangle_t tri = {{ geo->m_Tris[i*3+0],
					  geo->m_Tris[i*3+1],
					  geo->m_Tris[i*3+2] }};
		      border_tris.push_back(tri);
		    }

		  //convert quads to tris
		  for(unsigned int i = 0; i < geo->m_NumQuads; i++)
		    {
		      triangle_t tri0 = {{ geo->m_Quads[i*4 + 0],
					   geo->m_Quads[i*4 + 1],
					   geo->m_Quads[i*4 + 3] }};
		      triangle_t tri1 = {{ geo->m_Quads[i*4 + 1],
					   geo->m_Quads[i*4 + 2],
					   geo->m_Quads[i*4 + 3] }};
		      border_tris.push_back(tri0);
		      border_tris.push_back(tri1);
		    }
		}
	      
	      reference = reference.tri_surface();
	      
	      project_verts::project(border_verts.begin(),
				     border_verts.end(),

				     //border_tris.begin(),
				     //border_tris.end(),

				     reference.points.begin(),
				     reference.points.end(),
				     reference.tris.begin(),
				     reference.tris.end());

	      if(geo->m_GeoFrame)
		{
		  for(vector<unsigned int>::iterator i = border_vert_indices.begin();
		      i != border_vert_indices.end();
		      i++)
		    for(unsigned int j = 0; j < 3; j++)
		      geo->m_GeoFrame->verts[*i][j] = border_verts[i-border_vert_indices.begin()][j];
		}
	      else
		{
		  //all verts are boundary vertices so we don't need border_vert_indices
		  for(unsigned int i = 0; i < geo->m_NumPoints; i++)
		    for(unsigned int j = 0; j < 3; j++)
		      geo->m_Points[i*3+j] = border_verts[i][j];
		}
	    }

	  m_ZoomedOut->updateGL();
	  m_ZoomedIn->updateGL();
	}
      catch(exception& e)
	{
	  QMessageBox::critical(this,"Error: ",e.what());
	  return;
	}
    }
}

void NewVolumeMainWindow::exportZoomedInIsosurfaceSlot()
{
	/*
	QString filename = QFileDialog::getSaveFileName(QString::null, "IPoly files (*.ipoly)", this);

	if (!filename.isNull()) {
		IPolyRenderable ipr( m_SourceManager.getZoomedInVolume()->getMultiContour()->getIPoly());
		ipr.saveFile(filename);
	}
	
	QString filter;
	QString filename = QFileDialog::getSaveFileName(QString::null, "Rawn files (*.rawn);;Raw files (*.raw)", this, "Save As", "Save As", &filter);

	if (!filename.isNull()) {
		Geometry* geo = m_SourceManager.getZoomedInVolume()->getMultiContour()->getGeometry();
		if (!saveGeometry(filename, geo)) {
			QMessageBox::information(this, "Error Saving File", "There was an error saving the file");
		}
		delete geo;
	}
	*/
	GeometryLoader loader;
	Geometry* geo = m_ZoomedInRenderable.getMultiContour()->getGeometry();
	QString filter;
	QString filename;
	
	if (geo) {
		filename = Q3FileDialog::getSaveFileName(QString::null, (QString)(loader.getSaveFilterString().c_str()), this, "Save As", "Save As", &filter);
		
		if (!filename.isNull() && !loader.saveFile((std::string)(filename.ascii()), (std::string)(filter.ascii()), geo)) {
				QMessageBox::information(this, "Error Saving File", "There was an error saving the file");
		}
		delete geo;
	}
	else {
		QMessageBox::warning(this, tr("Export Subvolume Isosurface"),
				tr("There are no isosurfaces to export!"), 0,0);
	}
}

void NewVolumeMainWindow::exportZoomedOutIsosurfaceSlot()
{
	/*
	QString filename = QFileDialog::getSaveFileName(QString::null, "IPoly files (*.ipoly)", this);

	if (!filename.isNull()) {
		IPolyRenderable ipr( m_SourceManager.getZoomedOutVolume()->getMultiContour()->getIPoly());
		ipr.saveFile(filename);
	}
	
	QString filename = QFileDialog::getSaveFileName(QString::null, "Rawn files (*.rawn)", this);

	if (!filename.isNull()) {
		Geometry* geo = m_SourceManager.getZoomedOutVolume()->getMultiContour()->getGeometry();
		if (!saveGeometry(filename, geo)) {
			QMessageBox::information(this, "Error Saving File", "There was an error saving the file");
		}
		delete geo;
	}
	*/
	GeometryLoader loader;
	Geometry* geo = m_ThumbnailRenderable.getMultiContour()->getGeometry();
	QString filter;
	QString filename;
	
	if (geo) {
		filename = Q3FileDialog::getSaveFileName(QString::null, (QString)(loader.getSaveFilterString().c_str()), this, "Save As", "Save As", &filter);
	
		if (!filename.isNull() && !loader.saveFile((std::string)(filename.ascii()), (std::string)(filter.ascii()), geo)) {
			QMessageBox::information(this, "Error Saving File", "There was an error saving the file");
		}
		delete geo;
	}
	else {
		QMessageBox::warning(this, tr("Export Thumbnail Isosurface"),
			tr("There are no isosurfaces to export!"), 0,0);
	}
}

void NewVolumeMainWindow::resetGeometryTransformationSlot()
{
	// clear the transformation
	m_ThumbnailRenderable.getGeometryRenderer()->clearTransformation();
	// redraw
	m_ZoomedOut->updateGL();
}

void NewVolumeMainWindow::toggleGeometryTransformationSlot()
{
	// toggle the flag
	m_TransformGeometry = !m_TransformGeometry;
	// pass it along to the SimpleOpenGLWidget instance (right side only)
	m_ZoomedOut->setObjectManipulationMode(m_TransformGeometry);
}

Geometry* NewVolumeMainWindow::loadGeometry(const char* filename) const
{
	Geometry* geometry;

	FILE* fp;
	// open the file
	fp = fopen(filename, "r");
	if (!fp) {
		return 0;
	}

	// get the number of verts and triangles
	int numverts, numtris;
	if (2!=fscanf(fp, "%d %d", &numverts, &numtris)) {
		qDebug("Error reading in number of verts and tris");
		return 0;
	}

	// initialize the geometry
	geometry = new Geometry;
	geometry->AllocateTris(numverts, numtris);


	int c;
	// read in the verts
	for (c=0; c<numverts; c++) {
		// read in a single vert, which includes 
		// position and normal
		if (6!=fscanf(fp, "%f %f %f %f %f %f", 
			&(geometry->m_TriVerts[c*3+0]),
			&(geometry->m_TriVerts[c*3+1]),
			&(geometry->m_TriVerts[c*3+2]),
			&(geometry->m_TriVertNormals[c*3+0]),
			&(geometry->m_TriVertNormals[c*3+1]),
			&(geometry->m_TriVertNormals[c*3+2]))) {
			qDebug("Error reading in vert # %d", c);
			delete geometry;
			return 0;
		}
	}
	// read in the triangles
	for (c=0; c<numtris; c++) {
		// read in 3 integers for each triangle
		if (3!=fscanf(fp, "%d %d %d", 
			&(geometry->m_Tris[c*3+0]),
			&(geometry->m_Tris[c*3+1]),
			&(geometry->m_Tris[c*3+2]))) {
			qDebug("Error reading in tri # %d", c);
		}
		// check the bounds on each vert
		if ((int)geometry->m_Tris[c*3+0]>=numverts || (int)geometry->m_Tris[c*3+1]>=numverts || (int)geometry->m_Tris[c*3+2]>=numverts ) {
			qDebug("Bounds error reading in tri # %d", c);
			delete geometry;
			return 0;
		}
	}

	// tell the geometry object that we've already set up the normals
	geometry->SetTriNormalsReady();

	// close the file and return
	fclose(fp);
	return geometry;
}

bool NewVolumeMainWindow::saveGeometry(const char* filename, Geometry* geometry) const
{
	FILE* fp;
	// open the file
	fp = fopen(filename, "w");
	if (!fp) {
		return false;
	}

	// write the number of verts & tris
	if (0>=fprintf(fp, "%d %d\n", geometry->m_NumTriVerts, geometry->m_NumTris)) {
		qDebug("Error writing the number of verts and tris");
		return false;
	}

	int c;
	// write out the verts
	for (c=0; c<(int)geometry->m_NumTriVerts; c++) {
		if (0>=fprintf(fp, "%f %f %f %f %f %f\n", 
			(geometry->m_TriVerts[c*3+0]),
			(geometry->m_TriVerts[c*3+1]),
			(geometry->m_TriVerts[c*3+2]),
			(geometry->m_TriVertNormals[c*3+0]),
			(geometry->m_TriVertNormals[c*3+1]),
			(geometry->m_TriVertNormals[c*3+2]))) {
		//if (0>=fprintf(fp, "%f %f %f\n", 
		//	(geometry->m_TriVerts[c*3+0]),
		//	(geometry->m_TriVerts[c*3+1]),
		//	(geometry->m_TriVerts[c*3+2]))) {
			qDebug("Error writing out vert # %d", c);
			fclose(fp);
			return false;
		}
	}
	// write out the tris
	for (c=0; c<(int)geometry->m_NumTris; c++) {
		if (0>=fprintf(fp, "%d %d %d\n", 
			(geometry->m_Tris[c*3+0]),
			(geometry->m_Tris[c*3+1]),
			(geometry->m_Tris[c*3+2]))) {
			qDebug("Error writing out tri # %d", c);
			fclose(fp);
			return false;
		}
	}
	fclose(fp);
	return true;
}

QColor NewVolumeMainWindow::getSavedColor()
{
	// gets the color from the registry/settings file
	QSettings settings;
	settings.insertSearchPath(QSettings::Windows, "/CCV");

	bool result;
	QString colorString = settings.readEntry("/Volume Rover/BackgroundColor", QColor(0, 0, 0).name(), &result);
	return QColor(colorString);
}

void NewVolumeMainWindow::setSavedColor(const QColor& color)
{
	QSettings settings;
	settings.insertSearchPath(QSettings::Windows, "/CCV");
	settings.writeEntry("/Volume Rover/BackgroundColor", color.name());
}

QDir NewVolumeMainWindow::getCacheDir()
{
	QSettings settings;
	settings.insertSearchPath(QSettings::Windows, "/CCV");

	bool result;
	QString dirString = settings.readEntry("/Volume Rover/CacheDir", QString::null, &result);

	if (result) {
		// construct the directory object
		QDir cacheDir(dirString);
		// make sure the directory exists!
		// (if it's not there, Rover freaks out when you open a file)
		if (!cacheDir.exists()) {
			// create the cache directory
			cacheDir.mkdir(dirString);
		}
		// return the directory
		return QDir(dirString);
	}
	else {
	  // query the user
	  QDir dir;
	  QString s = Q3FileDialog::getExistingDirectory(dir.absPath(),
							this, "Choose the location of the cache directory",
							QString("Choose the location of the cache directory"), TRUE );
	  if ( !s.isNull() ) {
	    dir = QDir(s);
	  }

	  settings.writeEntry("/Volume Rover/CacheDir", dir.absPath());
	  return dir;
	}
}

QDir NewVolumeMainWindow::presentDirDialog(QDir defaultDir)
{
	// query the user
	QString s = Q3FileDialog::getExistingDirectory(
			defaultDir.absPath(),
			this, "Choose the location of the cache directory",
			QString("Choose the location of the cache directory"), TRUE );
	if ( !(s.isNull()) ) {
		return QDir(s);
	}
	else {
		return defaultDir;
	}
}

void NewVolumeMainWindow::recentFileSlot(int fileNum)
{
	QString file = m_RecentFiles.getFileName(fileNum);
	openFile(file);
}

inline unsigned int NewVolumeMainWindow::upToPowerOfTwo(unsigned int value) const
{
	unsigned int c = 0;
	unsigned int v = value;

	// round down to nearest power of two
	while (v>1) {
		v = v>>1;
		c++;
	}

	// if that isn't exactly the original value
	if ((v<<c)!=value) {
		// return the next power of two
		return (v<<(c+1));
	}
	else {
		// return this power of two
		return (v<<c);
	}
}

inline double NewVolumeMainWindow::texCoordOfSample(double sample, int bufferWidth, int canvasWidth, double bufferMin, double bufferMax) const
{
	// get buffer min and max in the texture's space
	double texBufferMin = 0.5 / (double)canvasWidth;
	double texBufferMax = ((double)bufferWidth - 0.5) / (double)canvasWidth;

	return (sample-bufferMin)/(bufferMax-bufferMin) * (texBufferMax-texBufferMin) + texBufferMin;
}

void NewVolumeMainWindow::copyToUploadableBufferDensity(RoverRenderable* roverRenderable, Extents* extents, unsigned int var)
{
	unsigned int dummy = 0;
	dummy = var;
	unsigned int j, k;
	unsigned int targetSlice, targetLine;
	unsigned int sourceSlice, sourceLine;

	VolumeBuffer* densityBuffer = roverRenderable->getVolumeBufferManager()->getVolumeBuffer(getVarNum());
	unsigned int widthX = densityBuffer->getWidth();
	unsigned int widthY = densityBuffer->getHeight();
	unsigned int widthZ = densityBuffer->getDepth();

	unsigned int canvasX = 0;
	canvasX = upToPowerOfTwo(widthX);
	unsigned int canvasY = 0;
	canvasY = upToPowerOfTwo(widthY);
	unsigned int canvasZ = 0;
	canvasZ = upToPowerOfTwo(widthZ);

	for (k=0; k<widthZ; k++) {
		targetSlice = (k)*canvasX*canvasY;
		sourceSlice = k*widthX*widthY;
		for (j=0; j<widthY; j++) {
			targetLine = (j)*canvasX;
			sourceLine = j*widthX;
			memcpy(m_UploadBuffer.get()+targetSlice+targetLine, densityBuffer->getBuffer()+sourceSlice+sourceLine, widthX);
		}
	}
}

void NewVolumeMainWindow::copyToUploadableBufferGradient(RoverRenderable* roverRenderable, Extents* extents, unsigned int var)

{
	unsigned int dummy = 0;
	dummy = var;
	unsigned int j, k;
	unsigned int targetSlice, targetLine;
	unsigned int sourceSlice, sourceLine;

	VolumeBuffer* gradBuffer = roverRenderable->getVolumeBufferManager()->getGradientBuffer(getVarNum());
	unsigned int widthX = gradBuffer->getWidth();
	unsigned int widthY = gradBuffer->getHeight();
	unsigned int widthZ = gradBuffer->getDepth();

	unsigned int canvasX = 0;
	canvasX = upToPowerOfTwo(widthX);
	unsigned int canvasY = 0;
	canvasY = upToPowerOfTwo(widthY);
	unsigned int canvasZ = 0;
	canvasZ = upToPowerOfTwo(widthZ);

	for (k=0; k<widthZ; k++) {
		targetSlice = (k)*canvasX*canvasY*4;
		sourceSlice = k*widthX*widthY*4;
		for (j=0; j<widthY; j++) {
			targetLine = (j)*canvasX*4;
			sourceLine = j*widthX*4;
			memcpy(m_UploadBuffer.get()+targetSlice+targetLine, gradBuffer->getBuffer()+sourceSlice+sourceLine, widthX*4);
		}
	}
}



void NewVolumeMainWindow::copyToUploadableBufferRGBA(RoverRenderable* roverRenderable, Extents* extents, unsigned int var, unsigned int offset)
{
	unsigned int j, k;
	unsigned int targetSlice, targetLine;
	unsigned int sourceSlice, sourceLine;

	VolumeBuffer* buffer = roverRenderable->getVolumeBufferManager()->getVolumeBuffer(var);
	unsigned int widthX = buffer->getWidth();
	unsigned int widthY = buffer->getHeight();
	unsigned int widthZ = buffer->getDepth();

	unsigned int canvasX = 0;
	canvasX = upToPowerOfTwo(widthX);
	unsigned int canvasY = 0;
	canvasY = upToPowerOfTwo(widthY);
	unsigned int canvasZ = 0;
	canvasZ = upToPowerOfTwo(widthZ);

	unsigned int c;
	for (k=0; k<widthZ; k++) {
		targetSlice = (k)*canvasX*canvasY*4;
		sourceSlice = k*widthX*widthY;
		for (j=0; j<widthY; j++) {
			targetLine = (j)*canvasX*4;
			sourceLine = j*widthX;
			for (c=0; c<widthX; c++) {
				m_UploadBuffer[targetSlice+targetLine+c*4+offset] = buffer->getBuffer()[sourceSlice+sourceLine+c];
			}
		}
	}
}

unsigned int NewVolumeMainWindow::getRedVariable() const
{
	if (m_SourceManager.hasSource()) {
		return m_RedBox->currentItem();
	}
	else {
		return 0;
	}
}

unsigned int NewVolumeMainWindow::getGreenVariable() const
{
	if (m_SourceManager.hasSource()) {
		return m_GreenBox->currentItem();
	}
	else {
		return 0;
	}
}

unsigned int NewVolumeMainWindow::getBlueVariable() const
{
	if (m_SourceManager.hasSource()) {
		return m_BlueBox->currentItem();
	}
	else {
		return 0;
	}
}

unsigned int NewVolumeMainWindow::getAlphaVariable() const
{
	if (m_SourceManager.hasSource()) {
		return m_AlphaBox->currentItem();
	}
	else {
		return 0;
	}
}

void NewVolumeMainWindow::updateRoverRenderable(RoverRenderable* roverRenderable, Extents* extents)
{
	if (m_RGBARendering) {
		double minX = extents->getXMin();
		double minY = extents->getYMin();
		double minZ = extents->getZMin();
		double maxX = extents->getXMax();
		double maxY = extents->getYMax();
		double maxZ = extents->getZMax();

		VolumeBuffer* redBuffer = roverRenderable->getVolumeBufferManager()->getVolumeBuffer(getRedVariable());
		VolumeBuffer* greenBuffer = roverRenderable->getVolumeBufferManager()->getVolumeBuffer(getGreenVariable());
		VolumeBuffer* blueBuffer = roverRenderable->getVolumeBufferManager()->getVolumeBuffer(getBlueVariable());
		VolumeBuffer* alphaBuffer = roverRenderable->getVolumeBufferManager()->getVolumeBuffer(getAlphaVariable());

		if(!redBuffer || !greenBuffer || !blueBuffer || !alphaBuffer)
		  {
		    //qDebug(QString("Error extracting volume: %1").arg(roverRenderable->getVolumeBufferManager()->getDownLoadManager()->error()));
		    qDebug("Error extracting volume.");
		    return;
		  }

		unsigned int canvasX = upToPowerOfTwo(redBuffer->getWidth());
		unsigned int canvasY = upToPowerOfTwo(redBuffer->getHeight());
		unsigned int canvasZ = upToPowerOfTwo(redBuffer->getDepth());

		// copy to uploadable buffer
		copyToUploadableBufferRGBA(roverRenderable, extents, getRedVariable(), 0);
		copyToUploadableBufferRGBA(roverRenderable, extents, getGreenVariable(), 1);
		copyToUploadableBufferRGBA(roverRenderable, extents, getBlueVariable(), 2);
		copyToUploadableBufferRGBA(roverRenderable, extents, getAlphaVariable(), 3);
		
		/* with border
		viewer->uploadColorMappedDataWithBorder(m_MainBuffer, canvasWidthX, canvasWidthY, canvasWidthZ);
		*/
		// without border
		QTime t;
		t.start();
		
		// upload to volume renderer
		roverRenderable->getVolumeRenderer()->uploadRGBAData(m_UploadBuffer.get(), canvasX, canvasY, canvasZ);
		qDebug("Time to upload : %d", t.elapsed());
		
		if (roverRenderable->getShadedVolumeRendering())
		{
			printf("Compute gradients and upload\n");
			roverRenderable->getVolumeRenderer()->calculateGradientsFromDensities(m_UploadBuffer.get(), canvasX, canvasY, canvasZ);
		}

		roverRenderable->setAspectRatio(fabs(maxX-minX), fabs(maxY-minY), fabs(maxZ-minZ));
		roverRenderable->getVolumeRenderer()->setTextureSubCube(
			texCoordOfSample(minX, redBuffer->getWidth(), canvasX, redBuffer->getMinX(), redBuffer->getMaxX()),
			texCoordOfSample(minY, redBuffer->getHeight(), canvasY, redBuffer->getMinY(), redBuffer->getMaxY()),
			texCoordOfSample(minZ, redBuffer->getDepth(), canvasZ, redBuffer->getMinZ(), redBuffer->getMaxZ()),
			texCoordOfSample(maxX, redBuffer->getWidth(), canvasX, redBuffer->getMinX(), redBuffer->getMaxX()),
			texCoordOfSample(maxY, redBuffer->getHeight(), canvasY, redBuffer->getMinY(), redBuffer->getMaxY()),
			texCoordOfSample(maxZ, redBuffer->getDepth(), canvasZ, redBuffer->getMinZ(), redBuffer->getMaxZ()));
		
		//qDebug("Done messing with volume viewer");

		roverRenderable->setShowVolumeRendering(true);
		
		// prepare multicontour
		// this is probably wrong if a border is being used
		// delete this stuff as soon as possible
		//contourManager->setData((unsigned char*)m_ThumbnailBuffer, widthX, widthY, widthZ,
		//	fabs(maxX-minX), fabs(maxY-minY), fabs(maxZ-minZ));
		roverRenderable->getMultiContour()->setData((unsigned char*)alphaBuffer->getBuffer(), 
			(unsigned char*)redBuffer->getBuffer(),
			(unsigned char*)greenBuffer->getBuffer(),
			(unsigned char*)blueBuffer->getBuffer(),
			redBuffer->getWidth(), redBuffer->getHeight(), redBuffer->getDepth(),
			fabs(maxX-minX), fabs(maxY-minY), fabs(maxZ-minZ),
			(minX-redBuffer->getMinX())/(redBuffer->getMaxX()-redBuffer->getMinX()),
			(minY-redBuffer->getMinY())/(redBuffer->getMaxY()-redBuffer->getMinY()),
			(minZ-redBuffer->getMinZ())/(redBuffer->getMaxZ()-redBuffer->getMinZ()),
			(maxX-redBuffer->getMinX())/(redBuffer->getMaxX()-redBuffer->getMinX()),
			(maxY-redBuffer->getMinY())/(redBuffer->getMaxY()-redBuffer->getMinY()),
			(maxZ-redBuffer->getMinZ())/(redBuffer->getMaxZ()-redBuffer->getMinZ()),
			minX, minY, minZ,
			maxX, maxY, maxZ);
		
		// prepare geometryRenderer
		// nothing to be done for geometryRenderer

	}
	else {
		// non-RGBA (colormapped) rendering



		double minX = extents->getXMin();
		double minY = extents->getYMin();
		double minZ = extents->getZMin();
		double maxX = extents->getXMax();
		double maxY = extents->getYMax();
		double maxZ = extents->getZMax();

		VolumeBuffer* densityBuffer = roverRenderable->getVolumeBufferManager()->getVolumeBuffer(getVarNum());
		if(!densityBuffer)
		  {
		    //qDebug(QString("Error extracting volume: %1").arg(roverRenderable->getVolumeBufferManager()->getDownLoadManager()->error()));
		    qDebug("Error extracting volume.");
		    return;
		  }

		unsigned int canvasX = upToPowerOfTwo(densityBuffer->getWidth());
		unsigned int canvasY = upToPowerOfTwo(densityBuffer->getHeight());
		unsigned int canvasZ = upToPowerOfTwo(densityBuffer->getDepth());

		// copy to uploadable buffer
		copyToUploadableBufferDensity(roverRenderable, extents, getVarNum());
		
		/* with border
		viewer->uploadColorMappedDataWithBorder(m_MainBuffer, canvasWidthX, canvasWidthY, canvasWidthZ);
		*/
		// without border
		QTime t;
		t.start();
		
		// upload to volume renderer
		roverRenderable->getVolumeRenderer()->uploadColorMappedData(m_UploadBuffer.get(), canvasX, canvasY, canvasZ);
		qDebug("Time to upload : %d", t.elapsed());
		
		// compute gradients and upload if shaded rendering is enabled
		if (roverRenderable->getShadedVolumeRendering())
		{
			printf("Compute gradients and upload\n");
			// calculate gradient on the fly.
			roverRenderable->getVolumeRenderer()->calculateGradientsFromDensities(m_UploadBuffer.get(), canvasX, canvasY, canvasZ);

			//t.restart();
			//roverRenderable->getVolumeRenderer()->calculateGradientsFromDensities(m_UploadBuffer, canvasX, canvasY, canvasZ);
			//qDebug("Time to compute gradients : %d", t.elapsed());
			//roverRenderable->getVolumeRenderer()->uploadGradients(const GLubyte* data, int width, int height, int depth);

			// we can reuse the canvasN variables
		//	copyToUploadableBufferGradient(roverRenderable, extents, getVarNum());
		//	roverRenderable->getVolumeRenderer()->uploadGradients(m_UploadBuffer.get(), canvasX, canvasY, canvasZ);
	//		Quaternion orientation = m_ZoomedOut->getViewInformation().getOrientation();
	//		float view[] = {(orientation)[0], (orientation)[1], (orientation)[2]};
	//		roverRenderable->getVolumeRenderer()->setView(view);
			//copyToUploadableBufferRGBA(roverRenderable, extents, getVarNum(), 3);
			//roverRenderable->getVolumeRenderer()->uploadRGBAData(m_UploadBuffer, canvasX, canvasY, canvasZ);
		}
		
		qDebug("Original Size: width: %d, height: %d, depth: %d", densityBuffer->getWidth(), densityBuffer->getHeight(), densityBuffer->getDepth());
		qDebug("Uploading data: width: %d, height: %d, depth: %d", canvasX, canvasY, canvasZ);
		
		/*
		qDebug("%f %f %f %f %f %f",texCoordOfSample(minX, densityBuffer->getWidth(), canvasX, densityBuffer->getMinX(), densityBuffer->getMaxX()),
			texCoordOfSample(minY, densityBuffer->getHeight(), canvasY, densityBuffer->getMinY(), densityBuffer->getMaxY()),
			texCoordOfSample(minZ, densityBuffer->getDepth(), canvasZ, densityBuffer->getMinZ(), densityBuffer->getMaxZ()),
			texCoordOfSample(maxX, densityBuffer->getWidth(), canvasX, densityBuffer->getMinX(), densityBuffer->getMaxX()),
			texCoordOfSample(maxY, densityBuffer->getHeight(), canvasY, densityBuffer->getMinY(), densityBuffer->getMaxY()),
			texCoordOfSample(maxZ, densityBuffer->getDepth(), canvasZ, densityBuffer->getMinZ(), densityBuffer->getMaxZ()));
		qDebug("%f %f %f %f %f %f\n",minX,minY,minZ,maxX,maxY,maxZ);
		*/

		roverRenderable->setAspectRatio(fabs(maxX-minX), fabs(maxY-minY), fabs(maxZ-minZ));
		roverRenderable->getVolumeRenderer()->setTextureSubCube(
			texCoordOfSample(minX, densityBuffer->getWidth(), canvasX, densityBuffer->getMinX(), densityBuffer->getMaxX()),
			texCoordOfSample(minY, densityBuffer->getHeight(), canvasY, densityBuffer->getMinY(), densityBuffer->getMaxY()),
			texCoordOfSample(minZ, densityBuffer->getDepth(), canvasZ, densityBuffer->getMinZ(), densityBuffer->getMaxZ()),
			texCoordOfSample(maxX, densityBuffer->getWidth(), canvasX, densityBuffer->getMinX(), densityBuffer->getMaxX()),
			texCoordOfSample(maxY, densityBuffer->getHeight(), canvasY, densityBuffer->getMinY(), densityBuffer->getMaxY()),
			texCoordOfSample(maxZ, densityBuffer->getDepth(), canvasZ, densityBuffer->getMinZ(), densityBuffer->getMaxZ()));
		
		//qDebug("Done messing with volume viewer");
		roverRenderable->setShowVolumeRendering(true);

		// prepare multicontour
		// this is probably wrong if a border is being used
		// delete this stuff as soon as possible
		//contourManager->setData((unsigned char*)m_ThumbnailBuffer, widthX, widthY, widthZ,
		//	fabs(maxX-minX), fabs(maxY-minY), fabs(maxZ-minZ));
		roverRenderable->getMultiContour()->setData((unsigned char*)densityBuffer->getBuffer(), densityBuffer->getWidth(), densityBuffer->getHeight(), densityBuffer->getDepth(),
			fabs(maxX-minX), fabs(maxY-minY), fabs(maxZ-minZ),
			(minX-densityBuffer->getMinX())/(densityBuffer->getMaxX()-densityBuffer->getMinX()),
			(minY-densityBuffer->getMinY())/(densityBuffer->getMaxY()-densityBuffer->getMinY()),
			(minZ-densityBuffer->getMinZ())/(densityBuffer->getMaxZ()-densityBuffer->getMinZ()),
			(maxX-densityBuffer->getMinX())/(densityBuffer->getMaxX()-densityBuffer->getMinX()),
			(maxY-densityBuffer->getMinY())/(densityBuffer->getMaxY()-densityBuffer->getMinY()),
			(maxZ-densityBuffer->getMinZ())/(densityBuffer->getMaxZ()-densityBuffer->getMinZ()),
			minX, minY, minZ,
			maxX, maxY, maxZ);
		
		// prepare geometryRenderer
		// nothing to be done for geometryRenderer
	}

}

void NewVolumeMainWindow::updateRecentlyUsedList(const QString& filename)
{
	m_RecentFiles.updateRecentFiles(filename);
}

void NewVolumeMainWindow::checkForConnection()
{
#ifdef USINGCORBA
	if (m_RenderServer) {
		RenderFrameAction->setEnabled(true);
		ServerSettingAction->setEnabled(true);
		DisconnectServerAction->setEnabled(true);
	}
	else {
		RenderFrameAction->setEnabled(false);
		ServerSettingAction->setEnabled(false);
		DisconnectServerAction->setEnabled(false);
	}
#else
	// disable the render servers
	RenderFrameAction->setEnabled(false);
	ServerSettingAction->setEnabled(false);
	DisconnectServerAction->setEnabled(false);
	ConnectServerAction->setEnabled(false);
#endif

}

bool NewVolumeMainWindow::checkError()
{
	if (m_ThumbnailRenderable.getVolumeBufferManager()->getDownLoadManager()->error()) {
		QMessageBox::critical( this, "Error", m_ThumbnailRenderable.getVolumeBufferManager()->getDownLoadManager()->errorReason() );
		m_ThumbnailRenderable.getVolumeBufferManager()->getDownLoadManager()->resetError();
		return true;
	}
	else if (m_ZoomedInRenderable.getVolumeBufferManager()->getDownLoadManager()->error()) {
		QMessageBox::critical( this, "Error", m_ZoomedInRenderable.getVolumeBufferManager()->getDownLoadManager()->errorReason() );
		m_ZoomedInRenderable.getVolumeBufferManager()->getDownLoadManager()->resetError();
		return true;
	}
	else {
		return false;
	}
}

void NewVolumeMainWindow::setUpdateMethod(VolumeSource::DownLoadFrequency method)
{
	switch (method) {
	case VolumeSource::DLFInteractive:
		m_UpdateMethod = UMInteractive;
		break;
	case VolumeSource::DLFDelayed:
		m_UpdateMethod = UMDelayed;
		break;
	case VolumeSource::DLFManual:
		m_UpdateMethod = UMManual;
		break;
	};
}

void NewVolumeMainWindow::updateVariableInfo( bool flush )
{
	if (flush) {
		// a new file has been loaded, reset the m_Saved* vars
		m_SavedDensityVar = 0;
		m_SavedTimeStep = 0;
	}
	else {
		// before updating anything, save the currently selected variable
		// (only if we're not in RGBA mode)
		if (m_VariableBox->isEnabled()) {
			m_SavedDensityVar = m_VariableBox->currentItem();
		}
		// and the current timestep
		if (m_TimeStep->isEnabled()) {
			m_SavedTimeStep = m_TimeStep->value();
		}
		else if (m_RGBATimeStep->isEnabled()) {
			m_SavedTimeStep = m_RGBATimeStep->value();
		}
	}
	
	// now update the variable toolbar
	if (m_RGBARendering) {
		if (m_SourceManager.hasSource()) {
			enableRGBAToolbar();
			disableDensityToolbar();
			m_VariableSelectionStack->raiseWidget(1);
		}
		else {
			disableRGBAToolbar();
			disableDensityToolbar();
		}
	}
	else {
		if (m_SourceManager.hasSource()) {
			enableDensityToolbar();
			disableRGBAToolbar();
			m_VariableSelectionStack->raiseWidget(0);
		}
		else {
			disableRGBAToolbar();
			disableDensityToolbar();
		}
	}

	// update the function min/max values for the colortable
	if (m_SourceManager.hasSource())
	{
		double min, max;
		if (m_VariableBox->isEnabled())
		{
			// colormapped rendering
			min = m_SourceManager.getSource()->getFunctionMinimum(m_SavedDensityVar, m_SavedTimeStep);
			max = m_SourceManager.getSource()->getFunctionMaximum(m_SavedDensityVar, m_SavedTimeStep);
		}
		else
		{
			// RGBA rendering; send the alpha variable's range
			min = m_SourceManager.getSource()->getFunctionMinimum(getAlphaVariable(), m_SavedTimeStep);
			max = m_SourceManager.getSource()->getFunctionMaximum(getAlphaVariable(), m_SavedTimeStep);
		}
		// tell the transfer function widget
		m_ColorTable->setDataMinMax(min,max);
	}
}

void NewVolumeMainWindow::enableDensityToolbar()
{
	//m_Toolbar->show();
	
	//enableVariableBox(m_VariableBox, 0);
	enableVariableBox(m_VariableBox, m_SavedDensityVar);
	m_VariableBox->setEnabled(true);
	m_TimeStep->setEnabled(true);
	m_TimeStep->setMaxValue(m_SourceManager.getSource()->getNumTimeSteps()-1);
	//m_TimeStep->setValue(0);
	m_TimeStep->setValue(m_SavedTimeStep);
}

void NewVolumeMainWindow::enableRGBAToolbar()
{
	//m_RGBAToolbar->show();
	enableVariableBox(m_RedBox, 0);
	enableVariableBox(m_GreenBox, 1);
	enableVariableBox(m_BlueBox, 2);
	enableVariableBox(m_AlphaBox, 3);
	m_RGBATimeStep->setEnabled(true);
	m_RGBATimeStep->setMaxValue(m_SourceManager.getSource()->getNumTimeSteps()-1);
	//m_RGBATimeStep->setValue(0);
	m_RGBATimeStep->setValue(m_SavedTimeStep);
}

void NewVolumeMainWindow::disableDensityToolbar()
{
	m_VariableBox->clear();
	m_VariableBox->setEnabled(false);
	m_TimeStep->setEnabled(false);
	//m_Toolbar->hide();
}

void NewVolumeMainWindow::disableRGBAToolbar()
{
	m_RedBox->clear();
	m_RedBox->setEnabled(false);
	m_GreenBox->clear();
	m_GreenBox->setEnabled(false);
	m_BlueBox->clear();
	m_BlueBox->setEnabled(false);
	m_AlphaBox->clear();
	m_AlphaBox->setEnabled(false);

	m_RGBATimeStep->setEnabled(false);
	//m_RGBAToolbar->hide();
}

void NewVolumeMainWindow::enableVariableBox(QComboBox* box, unsigned int var)
{
	unsigned int v;
	box->setEnabled(true);
	box->clear();
	for (v=0; v<m_SourceManager.getSource()->getNumVars(); v++) {
		box->insertItem(m_SourceManager.getSource()->getVariableName(v));
	}
	box->setCurrentItem(var%m_SourceManager.getNumVars());
}

void NewVolumeMainWindow::recordFrame()
{
	if (m_Animation) {
		//qDebug("recordFrame()");
		// query the current view state
		ViewInformation info = m_ZoomedOut->getViewInformation();
		info.setClipPlane((float)m_ThumbnailRenderable.getVolumeRenderer()->getNearPlane());
		// add a keyframe to the animation
		m_Animation->addKeyFrame(
			ViewState(info.getOrientation(), info.getTarget(),
								info.getWindowSize(), info.getClipPlane(), m_WireFrame),
			m_Time.elapsed());
	}
}

Terminal* NewVolumeMainWindow::getTerminal() const { return m_Terminal; }

#ifdef VOLUMEGRIDROVER
void NewVolumeMainWindow::gridRoverDepthChangedSlot(SliceCanvas *sc, int d)
{
  m_ThumbnailRenderable.setSlice(SliceRenderable::SliceAxis(sc->getSliceAxis()),d);
  m_ZoomedInRenderable.setSlice(SliceRenderable::SliceAxis(sc->getSliceAxis()),d);
#ifdef DETACHED_VOLUMEGRIDROVER
  m_ZoomedOut->updateGL();
  m_ZoomedIn->updateGL();
#endif
}
#endif

void NewVolumeMainWindow::receiveTilingGeometrySlot(const boost::shared_ptr<Geometry>& g)
{
  //send an event because this slot may be called in a non GUI thread
  QApplication::postEvent(this,new TilingInfoEvent("Received tiled geometry from VolumeGridRover!", g));
}

void NewVolumeMainWindow::showImage(const QImage& img)
{
  ImageViewer* iv = new ImageViewer();
  iv->setPixmap(QPixmap(img));
  iv->show();
  connect(this,SIGNAL(destroyed()),iv,SLOT(deleteLater()));
}

NewVolumeMainWindow::LocalSegThread::LocalSegThread(const char *filename, SegmentationDialog *dialog, NewVolumeMainWindow *nvmw, unsigned int stackSize)
 : QThread(this), m_NewVolumeMainWindow(nvmw)
   // : QThread(stackSize), m_NewVolumeMainWindow(nvmw) // old
{
	 type = dialog->m_SegTypeSelection->currentItem();
	 
	 m_Params[0] = std::string(filename);
	 
	 switch(dialog->m_SegTypeSelection->currentItem())
	 {
		 case 0: /* Capsid */
			 
			 switch(dialog->m_CapsidLayerType->currentItem())
			 {
				 case 0:
					 m_Params[1] = int(0);
					 m_Params[2] = dialog->m_TLowEditType0->text().toDouble();
					 m_Params[3] = dialog->m_X0EditType0->text().toInt();
					 m_Params[4] = dialog->m_Y0EditType0->text().toInt();
					 m_Params[5] = dialog->m_Z0EditType0->text().toInt();
					 /* necessary to make VC++6.0 happy */
					 m_Params[6] = XmlRpcValue(dialog->m_RunDiffusionType0->isChecked());
					 break;
				 case 1:
					 m_Params[1] = int(1);
					 m_Params[2] = dialog->m_TLowEditType1->text().toDouble();
					 m_Params[3] = dialog->m_X0EditType1->text().toInt();
					 m_Params[4] = dialog->m_Y0EditType1->text().toInt();
					 m_Params[5] = dialog->m_Z0EditType1->text().toInt();
					 m_Params[6] = dialog->m_X1EditType1->text().toInt();
					 m_Params[7] = dialog->m_Y1EditType1->text().toInt();
					 m_Params[8] = dialog->m_Z1EditType1->text().toInt();
					 m_Params[9] = XmlRpcValue(dialog->m_RunDiffusionType1->isChecked());
					 break;
				 case 2:
					 m_Params[1] = int(2);
					 m_Params[2] = dialog->m_TLowEditType2->text().toDouble();
					 m_Params[3] = dialog->m_SmallRadiusEditType2->text().toDouble();
					 m_Params[4] = dialog->m_LargeRadiusEditType2->text().toDouble();
					 break;
				 case 3:
					 m_Params[1] = int(3);
					 m_Params[2] = dialog->m_TLowEditType3->text().toDouble();
					 m_Params[3] = dialog->m_3FoldEditType3->text().toInt();
					 m_Params[4] = dialog->m_5FoldEditType3->text().toInt();
					 m_Params[5] = dialog->m_6FoldEditType3->text().toInt();
					 m_Params[6] = dialog->m_SmallRadiusEditType3->text().toDouble();
					 m_Params[7] = dialog->m_LargeRadiusEditType3->text().toDouble();
					 break;
			 }
			 
			 break;
		 case 1: /* Monomer */
			 
			 m_Params[1] = dialog->m_FoldNumEdit->text().toInt();
			 
			 break;
		 case 2: /* Subunit */
			 
			 m_Params[1] = dialog->m_HNumEdit->text().toInt();
			 m_Params[2] = dialog->m_KNumEdit->text().toInt();
			 m_Params[3] = dialog->m_3FoldEdit->text().toInt();
			 m_Params[4] = dialog->m_5FoldEdit->text().toInt();
			 m_Params[5] = dialog->m_6FoldEdit->text().toInt();
			 m_Params[6] = dialog->m_InitRadiusEdit->text().toInt();
			 
			 break;
		 case 3: /* Secondary structure detection */
			 m_Params[1] = dialog->m_HelixWidth->text().toDouble();
			 m_Params[2] = dialog->m_MinHelixWidthRatio->text().toDouble();
			 m_Params[3] = dialog->m_MaxHelixWidthRatio->text().toDouble();
			 m_Params[4] = dialog->m_MinHelixLength->text().toDouble();
			 m_Params[5] = dialog->m_SheetWidth->text().toDouble();
			 m_Params[6] = dialog->m_MinSheetWidthRatio->text().toDouble();
			 m_Params[7] = dialog->m_MaxSheetWidthRatio->text().toDouble();
			 m_Params[8] = dialog->m_SheetExtend->text().toDouble();
			 if (dialog->m_ThresholdCheck->isChecked())
			  	m_Params[9] = dialog->m_Threshold->text().toDouble();
			 else m_Params[9] = double( -1.0);
		 
		     break;
	 }
}

void NewVolumeMainWindow::LocalSegThread::run()
{
	using namespace XmlRpc; /* we're only going to use XmlRpcValue so we can make the code more uniform... */
	printf("NewVolumeMainWindow::LocalSegThread::run(): Local segmentation thread started.\n");
	XmlRpcValue result;
	
	switch(type)
	{
		case 0: /* Capsid */
			
			if(!virusSegCapsid(m_Params,result))
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFailedEvent("Error segmenting virus capsid!"));
			else
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFinishedEvent("Local segmentation complete."));
			
			break;
		case 1: /* Monomer */
			
			if(!virusSegMonomer(m_Params,result))
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFailedEvent("Error segmenting virus monomer!"));
			else
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFinishedEvent("Local segmentation complete."));
			
			break;
		case 2: /* Subunit */
		
			if(!virusSegSubunit(m_Params,result))
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFailedEvent("Error segmenting virus subunit!"));
			else
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFinishedEvent("Local segmentation complete."));

			break;
		case 3: /* Secondary structure detection */
		
			if(!secondaryStructureDetection(m_Params,result))
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFailedEvent("Error detecting secondary structure!"));
			else
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFinishedEvent("Local secondary structure detection complete."));
				
			break;
	}

	printf("NewVolumeMainWindow::LocalSegThread::run(): Local segmentation thread finished.\n");
}

NewVolumeMainWindow::RemoteSegThread::RemoteSegThread(SegmentationDialog *dialog, NewVolumeMainWindow *nvmw, unsigned int stackSize)
 : QThread(this), m_NewVolumeMainWindow(nvmw),m_XmlRpcClient(dialog->m_RemoteSegmentationHostname->text(), dialog->m_RemoteSegmentationPort->text().toInt())
   // : QThread(stackSize), m_NewVolumeMainWindow(nvmw),m_XmlRpcClient(dialog->m_RemoteSegmentationHostname->text(), dialog->m_RemoteSegmentationPort->text().toInt())
{
	 type = dialog->m_SegTypeSelection->currentItem();
	 
	 m_Params[0] = std::string(dialog->m_RemoteSegmentationFilename->text().ascii());
	 
	 switch(dialog->m_SegTypeSelection->currentItem())
	 {
		 case 0: /* Capsid */
			 
			 switch(dialog->m_CapsidLayerType->currentItem())
			 {
				 case 0:
					 m_Params[1] = int(0);
					 m_Params[2] = dialog->m_TLowEditType0->text().toDouble();
					 m_Params[3] = dialog->m_X0EditType0->text().toInt();
					 m_Params[4] = dialog->m_Y0EditType0->text().toInt();
					 m_Params[5] = dialog->m_Z0EditType0->text().toInt();
					 m_Params[6] = XmlRpcValue(dialog->m_RunDiffusionType0->isChecked());
					 break;
				 case 1:
					 m_Params[1] = int(1);
					 m_Params[2] = dialog->m_TLowEditType1->text().toDouble();
					 m_Params[3] = dialog->m_X0EditType1->text().toInt();
					 m_Params[4] = dialog->m_Y0EditType1->text().toInt();
					 m_Params[5] = dialog->m_Z0EditType1->text().toInt();
					 m_Params[6] = dialog->m_X1EditType1->text().toInt();
					 m_Params[7] = dialog->m_Y1EditType1->text().toInt();
					 m_Params[8] = dialog->m_Z1EditType1->text().toInt();
					 m_Params[9] = XmlRpcValue(dialog->m_RunDiffusionType1->isChecked());
					 break;
				 case 2:
					 m_Params[1] = int(2);
					 m_Params[2] = dialog->m_TLowEditType2->text().toDouble();
					 m_Params[3] = dialog->m_SmallRadiusEditType2->text().toDouble();
					 m_Params[4] = dialog->m_LargeRadiusEditType2->text().toDouble();
					 break;
				 case 3:
					 m_Params[1] = int(3);
					 m_Params[2] = dialog->m_TLowEditType3->text().toDouble();
					 m_Params[3] = dialog->m_3FoldEditType3->text().toInt();
					 m_Params[4] = dialog->m_5FoldEditType3->text().toInt();
					 m_Params[5] = dialog->m_6FoldEditType3->text().toInt();
					 m_Params[6] = dialog->m_SmallRadiusEditType3->text().toDouble();
					 m_Params[7] = dialog->m_LargeRadiusEditType3->text().toDouble();
					 break;
			 }
			 
			 break;
		 case 1: /* Monomer */
			 
			 m_Params[1] = dialog->m_FoldNumEdit->text().toInt();
			 
			 break;
		 case 2: /* Subunit */
			 
			 m_Params[1] = dialog->m_HNumEdit->text().toInt();
			 m_Params[2] = dialog->m_KNumEdit->text().toInt();
			 m_Params[3] = dialog->m_3FoldEdit->text().toInt();
			 m_Params[4] = dialog->m_5FoldEdit->text().toInt();
			 m_Params[5] = dialog->m_6FoldEdit->text().toInt();
			 m_Params[6] = dialog->m_InitRadiusEdit->text().toInt();
			 
			 break;
		 case 3: /* Secondary structure detection */
			 m_Params[1] = dialog->m_HelixWidth->text().toDouble();
			 m_Params[2] = dialog->m_MinHelixWidthRatio->text().toDouble();
			 m_Params[3] = dialog->m_MaxHelixWidthRatio->text().toDouble();
			 m_Params[4] = dialog->m_MinHelixLength->text().toDouble();
			 m_Params[5] = dialog->m_SheetWidth->text().toDouble();
			 m_Params[6] = dialog->m_MinSheetWidthRatio->text().toDouble();
			 m_Params[7] = dialog->m_MaxSheetWidthRatio->text().toDouble();
			 m_Params[8] = dialog->m_SheetExtend->text().toDouble();
		 
		     break;
	 }
}

void NewVolumeMainWindow::RemoteSegThread::run()
{
	using namespace XmlRpc; /* we're only going to use XmlRpcValue so we can make the code more uniform... */
	printf("NewVolumeMainWindow::RemoteSegThread::run(): Remote segmentation thread started.\n");
	XmlRpcValue result;
	
	switch(type)
	{
		case 0: /* Capsid */
			
			if(!m_XmlRpcClient.execute("SegmentCapsid",m_Params,result))
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFailedEvent("Error segmenting virus capsid!"));
			else
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFinishedEvent("Remote segmentation complete."));
			
			break;
		case 1: /* Monomer */
			
			if(!m_XmlRpcClient.execute("SegmentMonomer",m_Params,result))
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFailedEvent("Error segmenting virus monomer!"));
			else
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFinishedEvent("Remote segmentation complete."));
			
			break;
		case 2: /* Subunit */
		
			if(!m_XmlRpcClient.execute("SegmentSubunit",m_Params,result))
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFailedEvent("Error segmenting virus subunit!"));
			else
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFinishedEvent("Remote segmentation complete."));

			break;
		case 3: /* Secondary structure detection */
		
			if(!m_XmlRpcClient.execute("SecondaryStructureDetection",m_Params,result))
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFailedEvent("Error detecting secondary structure!"));
			else
				QApplication::postEvent(m_NewVolumeMainWindow,new SegmentationFinishedEvent("Remote secondary structure detection complete."));

			break;
	}

	printf("NewVolumeMainWindow::RemoteSegThread::run(): Remote segmentation thread finished.\n");
}

#ifdef USING_PE_DETECTION
NewVolumeMainWindow::PEDetectionThread::PEDetectionThread(const QString& filename,
							  PEDetectionDialog *dialog, 
							  NewVolumeMainWindow *nvmw, 
							  unsigned int stackSize)
  : QThread(stackSize), m_NewVolumeMainWindow(nvmw),
    m_XmlRpcClient(dialog->m_Hostname->text(), dialog->m_Port->text().toInt()),
    m_Filename(filename), m_Remote(dialog->m_RunRemotely->isChecked())
{
  if(m_Remote) m_Filename = dialog->m_RemoteFile->text();
}

void NewVolumeMainWindow::PEDetectionThread::run()
{
  if(m_Remote)
    {
      XmlRpcValue params = std::string(m_Filename.ascii()), result;
      if(!m_XmlRpcClient.execute("PulmonaryEmbolusDetection",params,result))
	QApplication::postEvent(m_NewVolumeMainWindow,
				new PEDetectionFailedEvent("Error running Pulmonary Embolus Detection remotely"));
      else
	QApplication::postEvent(m_NewVolumeMainWindow,
				new PEDetectionFinishedEvent("Remote Pulmonary Embolus Detection finished"));
    }
  else
    {
      PEDetection(m_Filename.ascii());
      QApplication::postEvent(m_NewVolumeMainWindow,
			      new PEDetectionFinishedEvent("Local Pulmonary Embolus Detection finished"));
    }
}
#endif

#ifdef USING_POCKET_TUNNEL
NewVolumeMainWindow::PocketTunnelThread::PocketTunnelThread(Geometry *inputMesh,
							    NewVolumeMainWindow *nvmw,
							    unsigned int stackSize)
  : QThread(stackSize), m_NewVolumeMainWindow(nvmw), m_InputMesh(inputMesh)
{
}

void NewVolumeMainWindow::PocketTunnelThread::run()
{
  Geometry *newgeom;

  newgeom = PocketTunnel::pocket_tunnel_fromsurf(m_InputMesh);
  if(newgeom)
    QApplication::postEvent(m_NewVolumeMainWindow,
			    new PocketTunnelFinishedEvent("Pocket Tunnel finished sucessfully.",newgeom));
  else
    QApplication::postEvent(m_NewVolumeMainWindow,
			    new PocketTunnelFailedEvent("Pocket Tunnel failed!"));
}
#endif

void NewVolumeMainWindow::MSLevelSetSlot()
{
#ifdef USING_MSLEVELSET
  using namespace MumfordShahLevelSet;

  if(!m_SourceManager.hasSource()) {
    QMessageBox::warning(this, tr("MS Level Set"),
			 tr("A volume must be loaded for this feature to work."), 0,0);
    return;
  }

  if(m_RGBARendering)
    {
      QMessageBox::warning(this, tr("MS Level Set"),
			   tr("This feature is not available for RGBA volumes."), 0,0);
      return;
    }

  VolMagick::Volume vol;
  float *data;
  MSLevelSetDialog dialog(this);

  MSLevelSet mslsSolver;
  MSLevelSetParams MSLSParams;
  MSLSParams.lambda1 = 1.0; MSLSParams.lambda2 = 1.0;
  MSLSParams.mu = 32.5125; MSLSParams.nu = 0.0;
  MSLSParams.deltaT = 0.01; MSLSParams.epsilon = 1.0;
  MSLSParams.nIter = 15; MSLSParams.medianIter = 10;
  MSLSParams.DTWidth = 10; MSLSParams.medianTolerance = 0.0005f;
  MSLSParams.subvolDim = 128;
  MSLSParams.PDEBlockDim = 4; MSLSParams.avgBlockDim = 4;
  MSLSParams.SDTBlockDim = 4; MSLSParams.medianBlockDim = 4;
  MSLSParams.superEllipsoidPower = 8.0;
  MSLSParams.BBoxOffset = 5;
  MSLSParams.init_interface_method = 0;

  dialog.paramReference(&MSLSParams);//Pass the struct pointer to the dialog object. This must be done before exec()
  if(dialog.exec() != QDialog::Accepted)
    return;

  /*
    Check if we are doing in place filtering of the actual volume data file
    instead of simply the current subvolume buffer
    TODO: need to implement out-of-core filtering using VolMagick
  */
  if(!dialog.m_Preview->isChecked())
    {
      try
	{
	  VolMagick::readVolumeFile(vol,
				    m_VolumeFileInfo.filename(),
				    getVarNum(),
				    getTimeStep());
	  vol.voxelType(VolMagick::Float); //forces float voxel type
	  data = reinterpret_cast<float*>(*vol);
	  if(mslsSolver.runSolver(data, vol.XDim(), vol.YDim(), vol.ZDim(), &MSLSParams))
	    {
	      vol.desc("Mumford-Shah Level Set");
	      QDir tmpdir(m_CacheDir.absPath() + "/tmp");
	      if(!tmpdir.exists()) tmpdir.mkdir(tmpdir.absPath());
	      QFileInfo tmpfile(tmpdir,"tmp.rawv");
	      qDebug("Creating volume %s",tmpfile.absFilePath().ascii());
	      QString newfilename = QDir::convertSeparators(tmpfile.absFilePath());
	      VolMagick::createVolumeFile(newfilename,
					  vol.boundingBox(),
					  vol.dimension(),
					  std::vector<VolMagick::VoxelType>(1, vol.voxelType()));
	      VolMagick::writeVolumeFile(vol,newfilename);
	      openFile(newfilename);
	    }
	  else
	    QMessageBox::warning(this,
				 "MS Level Set",
				 "Solver failed to execute!", 0, 0);
	}
      catch(std::exception& e)
	{
	  QMessageBox::critical(this,"Error: ",e.what());
	  return;
	}

      return;
    }

#ifdef USING_MSLEVELSET_SINGLETHREAD
  explorerChangedSlot();

  VolumeBuffer *densityBuffer;
  unsigned int xdim, ydim, zdim, i, len;
  unsigned char *volume;
  
  densityBuffer = m_ZoomedInRenderable.getVolumeBufferManager()->getVolumeBuffer(getVarNum());
  volume = (unsigned char *)densityBuffer->getBuffer();
  xdim = densityBuffer->getWidth();
  ydim = densityBuffer->getHeight();
  zdim = densityBuffer->getDepth();
  len = xdim*ydim*zdim;
  
  VolMagick::Voxels vox(VolMagick::Dimension(xdim,ydim,zdim),VolMagick::UChar);
  memcpy(*vox,volume,len*sizeof(unsigned char));
  vox.voxelType(VolMagick::Float); //the solver wants float
  data = reinterpret_cast<float*>(*vox);
  if(!mslsSolver.runSolver(data, vox.XDim(), vox.YDim(), vox.ZDim(), &MSLSParams))
  {
    QMessageBox::critical(this,"Error","MSLevelSet Solver failed to execute!");
    return;
  }
  vox.voxelType(VolMagick::UChar); //convert back to unsigned char for the subvol buffer
  memcpy(volume,*vox,len*sizeof(unsigned char));
  
  m_ZoomedIn->makeCurrent();
  updateRoverRenderable(&m_ZoomedInRenderable, &m_ZoomedInExtents);
printf("NewVolumeMainWindow::MSLevelSetSlot()\n");

  m_ZoomedIn->updateGL();
#else
  if(m_MSLevelSetThread && m_MSLevelSetThread->running())
    QMessageBox::critical(this,"Error","MSLevelSet thread already running!");
  else
    {
      explorerChangedSlot();
      
      VolumeBuffer *densityBuffer;
      unsigned int xdim, ydim, zdim, i, len;
      unsigned char *volume;
      
      densityBuffer = m_ZoomedInRenderable.getVolumeBufferManager()->getVolumeBuffer(getVarNum());
      volume = (unsigned char *)densityBuffer->getBuffer();
      xdim = densityBuffer->getWidth();
      ydim = densityBuffer->getHeight();
      zdim = densityBuffer->getDepth();
      len = xdim*ydim*zdim;
      
      VolMagick::Volume vol(VolMagick::Dimension(xdim,ydim,zdim),
			    VolMagick::UChar,
			    VolMagick::BoundingBox(m_ZoomedInExtents.getXMin(), 
						   m_ZoomedInExtents.getYMin(), 
						   m_ZoomedInExtents.getZMin(),
						   m_ZoomedInExtents.getXMax(),
						   m_ZoomedInExtents.getYMax(),
						   m_ZoomedInExtents.getZMax()));
      memcpy(*vol,volume,len*sizeof(unsigned char));
      if(m_MSLevelSetThread) delete m_MSLevelSetThread;
      m_MSLevelSetThread = new MSLevelSetThread(MSLSParams,vol);
      m_MSLevelSetThread->start();
    }
#endif

#else
  QMessageBox::information(this, tr("MSLevelSet"),
			   tr("Mumford-Shah Level Set solver not built into volrover!"), QMessageBox::Ok);
#endif
}

#ifdef USING_MSLEVELSET
NewVolumeMainWindow::MSLevelSetThread::MSLevelSetThread(const MSLevelSetParams& params,
							const VolMagick::Volume& vol, 
							unsigned int stackSize)
  : QThread(stackSize), m_Params(params), 
    m_VolData(vol)
{
  m_VolData.voxelType(VolMagick::Float); //the solver wants float

  //force a local copy to remove potential race condiditon on internal volume data
  VolMagick::Volume localvol(m_VolData.dimension(),
			     m_VolData.voxelType(),
			     m_VolData.boundingBox());
  memcpy(*localvol,
	 *m_VolData,
	 m_VolData.XDim()*m_VolData.YDim()*m_VolData.ZDim()*sizeof(float));
  m_VolData = localvol;
}

void NewVolumeMainWindow::MSLevelSetThread::run()
{
  using namespace MumfordShahLevelSet;

  MSLevelSet mslsSolver;
  if(!mslsSolver.runSolver(reinterpret_cast<float*>(*m_VolData),
			   m_VolData.XDim(),
			   m_VolData.YDim(),
			   m_VolData.ZDim(),
			   &m_Params,
			   sendUpdate,
			   this))
  {
    QApplication::postEvent(qApp->mainWidget(),
			    new MSLevelSetFailedEvent("MSLevelSet solver failed to execute!"));
    return;
  }

  sendUpdate(reinterpret_cast<float*>(*m_VolData),
	     m_VolData.XDim(),
	     m_VolData.YDim(),
	     m_VolData.ZDim(),
	     this);
}

void NewVolumeMainWindow::MSLevelSetThread::sendUpdate(const float* vol,
						       int dimx,
						       int dimy,
						       int dimz,
						       void *context)
{
  if(context == NULL)
    {
      QApplication::postEvent(qApp->mainWidget(),
			      new MSLevelSetFailedEvent("No context, unable to send update!!"));
      return;
    }

  MSLevelSetThread *self = reinterpret_cast<MSLevelSetThread*>(context);

  boost::scoped_array<float> newdata(new float[dimx*dimy*dimz]);
  memcpy(newdata.get(),vol,dimx*dimy*dimz*sizeof(float));
  memcpy(*self->m_VolData,vol,dimx*dimy*dimz*sizeof(float));

  //extract a contour for the level set
  self->m_ContourExtractor.setVolume(self->m_VolData);
  FastContouring::TriSurf contour =
    self->m_ContourExtractor.extractContour(0.0);

  boost::shared_ptr<Geometry> geom(new Geometry());
  geom->AllocateTris(contour.verts.size()/3,
		     contour.tris.size()/3);
  geom->AllocateTriVertColors();
  memcpy(geom->m_Points.get(),
	 &(contour.verts[0]),
	 contour.verts.size()*sizeof(float));
  memcpy(geom->m_PointColors.get(),
	 &(contour.colors[0]),
	 contour.colors.size()*sizeof(float));
  memcpy(geom->m_Tris.get(),
	 &(contour.tris[0]),
	 contour.tris.size()*sizeof(unsigned int));
  
  QApplication::postEvent(qApp->mainWidget(),
			  new MSLevelSetFinishedEvent("*************** MSLevelSet iteration finished! ***************",
						      self->m_VolData,
						      geom));
}
#endif


#ifdef USING_RECONSTRUCTION
///Reconstruction from projections

bool NewVolumeMainWindow::ReconstructionFromProjectionSlot()
{
  //qDebug("test for Reconstruction.");
  int i,j,k, bN, gM, m,n, N;
  int nx,ny,nz, recon_method;
  double c1,c2,c3, minv,maxv, Volume=0.0;
  //float *voxelvalues, *coordinates,*coefficents;
  int   iter, phantom, tolnv, newnv, bandwidth, ordermanner, flow, thickness;
  double rot, tilt, psi;
  double  reconj1, alpha, fac, tau, al, be, ga, la; 
  double  Al, Be, Ga, La;
  EulerAngles *eulers=NULL ;


  long double *SchmidtMat = NULL;
  double rotmat[9], translate[3]={0,0,0};
  Oimage *Object = NULL, *image = NULL;
  boost::tuple<bool,VolMagick::Volume> Result;
  const char *name, *path;
  const char *name1,*path1;



  /*

  if(m_Geometries.getNumberOfRenderables() == 0) 
  { 
  QMessageBox::information(this, "High Level Set Reconstruction", "Load a geometry file first."); 
  return false; 
  }
  */
  ReconstructionImpl dialog(this, 0, true);
  QDir dir;
  //dialog.m_lineEditDir->setText(dir.absPath());

  dialog.m_ImageDim->setValidator(new QDoubleValidator(this));
  dialog.m_SplineDim->setValidator(new QDoubleValidator(this));
 
  dialog.m_Rot->setValidator(new QDoubleValidator(this));
  dialog.m_Tilt->setValidator(new QDoubleValidator(this));
  dialog.m_Psi->setValidator(new QDoubleValidator(this));


  dialog.m_DeltaFuncAlpha->setValidator(new QDoubleValidator(this));
  dialog.m_NarrowBandSub->setValidator(new QDoubleValidator(this));

  dialog.m_MolVolume->setValidator(new QDoubleValidator(this));


  dialog.m_IterNumber->setValidator(new QDoubleValidator(this));
  dialog.m_TimeStep->setValidator(new QDoubleValidator(this));

  dialog.m_Reconj1->setValidator(new QDoubleValidator(this));
  dialog.m_Reconal->setValidator(new QDoubleValidator(this));
  dialog.m_Reconbe->setValidator(new QDoubleValidator(this));
  dialog.m_Reconga->setValidator(new QDoubleValidator(this));
  dialog.m_Reconla->setValidator(new QDoubleValidator(this));


  dialog.m_NewNv->setValidator(new QDoubleValidator(this));
  dialog.m_BandWidth->setValidator(new QDoubleValidator(this));


  dialog.m_PhantomId->setValidator(new QDoubleValidator(this));
  dialog.m_Flow->setValidator(new QDoubleValidator(this));
  dialog.m_Thickness->setValidator(new QDoubleValidator(this));


  if(dialog.exec() == QDialog::Accepted)
    {
      QDir result(dialog.m_lineEditDir->text());
      QDir result1(dialog.m_LoadInitF->text());
      
      m_Itercounts = m_Itercounts + 1;
      printf("\nm_Itercounts=%d ", m_Itercounts); //getchar();

      switch(dialog.m_buttonGroup_2->selectedId())
        {
        case 0:
          name = dialog.m_lineEditDir->text().ascii();
          result.cdUp();
          path = result.path().ascii();
          printf("\nname=%s reslut.path=%s \n", name, path) ; //very important.  05-25-2009.
          tolnv = atoi(dialog.m_TotalNum->text().ascii());

          name1 = dialog.m_LoadInitF->text().ascii();
          result1.cdUp();
          path1 = result1.path().ascii();
          printf("\nname1=%s reslut1.path1=%s \n", name1, path1) ; //very important.  05-25-2009.
          break;

        case 1:
          rot  = atof(dialog.m_Rot->text().ascii());
          tilt = atof(dialog.m_Tilt->text().ascii());
          psi  = atof(dialog.m_Psi->text().ascii());
          tolnv = (int)rot*tilt;
          phantom = atoi(dialog.m_PhantomId->text().ascii());
          break;
        }


      //rot  = atof(dialog.m_Rot->text().ascii());
      //tilt = atof(dialog.m_Tilt->text().ascii());
      //psi  = atof(dialog.m_Psi->text().ascii());

      //speed up parameters.
      ordermanner = dialog.m_OrderCombo->currentItem();
      newnv    = atoi(dialog.m_NewNv->text().ascii());
      bandwidth= atoi(dialog.m_BandWidth->text().ascii());

      //fixed parameters: narrow band parameters and flow.
      alpha   = atof(dialog.m_DeltaFuncAlpha->text().ascii());
      fac     = atof(dialog.m_NarrowBandSub->text().ascii());
      Volume  = atof(dialog.m_MolVolume->text().ascii()); 

      //unfixed parameters.
      iter  = atoi(dialog.m_IterNumber->text().ascii());
      tau   = atof(dialog.m_TimeStep->text().ascii());
      flow    = atoi(dialog.m_Flow->text().ascii());
      thickness    = atoi(dialog.m_Thickness->text().ascii());

      reconj1 = atof(dialog.m_Reconj1->text().ascii());
      al = atof(dialog.m_Reconal->text().ascii());
      be = atof(dialog.m_Reconbe->text().ascii());
      ga = atof(dialog.m_Reconga->text().ascii());
      la = atof(dialog.m_Reconla->text().ascii());
	
  

      //for test phantom.
      n = atoi(dialog.m_ImageDim->text().ascii());
      m = atoi(dialog.m_SplineDim->text().ascii());

 
      //if(abs(n%2) > 0.0 || abs(m%2) > 0.0 ) {printf("\nn,m should be even integars.");return false;} 

      ///////phantom = dialog.Phantom_Combo->currentItem();
        //phantom = atoi(dialog.m_PhantomId->text().ascii());
        //printf("parameters are \n DeltaFuncAlpha = %f,  Volume = %f \nIter Number  = %d,tau = %f, \nJ1 al be ga la = %f %f %f %f %f  \nAl Be Ga La = %f %f %f %f \nphantom=%d rot=%f tilt=%f psi=%f \nImage Dim=%d SplineBox=%d", 
	//		 alpha, Volume, iter, tau, reconj1, al, be, ga, la, Al,Be, Ga, La, phantom, rot, tilt, psi, n,m);
        //	  getchar();


               
        //if ( alpha == 0 ) {printf("\nerror. alpha should not be equal to zero.\n");return false;}
        /*
          VolMagick::Volume Vol;
          VolMagick::readVolumeFile(Vol,m_VolumeFileInfo.filename(),getVarNum(),getTimeStep());
          ddim[0] = Vol.XDim();
          ddim[1] = Vol.YDim();
          ddim[2] = Vol.ZDim();

          TotalSize = ddim[0] * ddim[1] *ddim[2];

          voxelvalues = (float *)malloc(ddim[0]*ddim[1]*ddim[2]*sizeof(float));
          coordinates = (float *)malloc(3*ddim[0]*ddim[1]*ddim[2]*sizeof(float));
          coefficents  = (float *)malloc(ddim[0]*ddim[1]*ddim[2]*sizeof(float));
          int numbpts;
          float weightc[3];

          numbpts = 0;
        */


        //The data is listed with the z co-ordinate varying fastest, and x varying slowest.

        /*  for(VolMagick::uint64 i = 0; i < ddim[0]; i++)
            for(VolMagick::uint64 j = 0; j < ddim[1]; j++)
            for(VolMagick::uint64 k = 0; k < ddim[2]; k++) {
            voxelvalues[((i*ddim[1]+j)*ddim[2]+k)] = Vol(i,j,k);
            // printf("\n voxelvalues = %f ", Vol(i,j,k));
            coordinates[3*((i*ddim[1]+j)*ddim[2]+k)+0] = Vol.XMin() + k * Vol.XSpan();
            coordinates[3*((i*ddim[1]+j)*ddim[2]+k)+1] = Vol.XMin() + j * Vol.XSpan();
            coordinates[3*((i*ddim[1]+j)*ddim[2]+k)+2] = Vol.XMin() + i * Vol.XSpan();
       
            //  printf("\n vertex_position %f %f %f ", x,y,z);
       
  
            }

            printf("ddim = %d %d %d size = %d \n", ddim[0], ddim[1], ddim[2],ddim[0]*ddim[1]*ddim[2]);
        */



        N  = n;                    //0928.
        printf("\ndim ==================================%d ", N);
        if(n%2 !=0 ) n = n -1;                                           //0928.
        nx = n; //n is the image size.
        ny = n;
        nz = n;


        //Views.
        //nview= ObtainViewVectorFromTriangularSphere("../Reconstruction/icosa.raw");

        /*
          nview = (Views *)malloc(sizeof(Views));
          nview->next = NULL;

          nview->x =1.0;//1/sqrt(2);
          nview->y = 0.0;//1/sqrt(2);
          nview->z = 0.0;//1/sqrt(2);
          nview->a = 0.0;// PI;
        */



        /*
          for ( i = 0; i < (int)(rot*tilt); i++ )
          {printf("\nrot=%f tilt=%f psi=%f ", eulers->rot, eulers->tilt, eulers->psi); eulers=eulers->next;}
          getchar();
        */



        if(m_Itercounts == 1)
          { 

            recon_method = 2;

            if(dialog.m_Parseval->isChecked() == 1) recon_method = 1;
            if(dialog.m_RealSpace->isChecked()== 1) recon_method = 2;

            // if ( dialog.m_LoadInitF->isEnabled()==1) printf("\n Load Init f.");
            //  reconstruction->InitialFunction(0,name1,path1);
            reconstruction->setOrderManner(ordermanner);
  
            Default_newnv     = newnv;
            Default_bandwidth = bandwidth;
            Default_flow      = flow; 
            Default_thickness   = thickness;

            reconstruction->Initialize(nx,ny,nz,m,newnv, bandwidth, alpha, fac, Volume, flow, recon_method);
            reconstruction->setThick(thickness);

            reconstruction->setTolNvRmatGdSize(tolnv);
            reconstruction->SetJ12345Coeffs(reconj1, al, be, ga, la);
            if ( dialog.m_LoadInitF->isEnabled()==0) {name1 = NULL;path1 = NULL;}  
            reconstruction->InitialFunction(0,name1,path1);

            reconManner = dialog.m_buttonGroup_2->selectedId();
            printf("\nreconManner================%d ", reconManner);



            if(dialog.m_RemoteReconRun->isChecked() == 1)

              {

                std::cout<<"\n Begin reconstruction on server \n";

                using namespace XmlRpc;

                XmlRpc::XmlRpcValue m_Params, result;


                m_Params[0] = reconManner;
                m_Params[1] = iter;
                m_Params[2] = tau;
                //  	m_Params[3] = eulers;
                m_Params[4] = m_Itercounts;
                m_Params[5] = phantom;

                m_Params[6] = thickness;
                m_Params[7] = flow;
                m_Params[8] = bandwidth;
                m_Params[9] = newnv;
                m_Params[10] = reconj1;
                m_Params[11] = al;
                m_Params[12] = be;
                m_Params[13] = ga;
                m_Params[14] = la;

                m_Params[15] = name;
                m_Params[16] = path;
                m_Params[17] = N; // N is dimension( nx, ny)

                m_Params[18] = rot;
                m_Params[19] = tilt;
                m_Params[20] = psi;

                m_Params[21] = ordermanner;
                m_Params[22] = nx;
                m_Params[23] = ny;
                m_Params[24] = nz;
                m_Params[25] = m;
                m_Params[26] = alpha;
                m_Params[27] = fac;
                m_Params[28] = Volume;

                //	m_Params[29] = flow;
                m_Params[30] = recon_method;
                m_Params[31] = tolnv;

                /*  if ( dialog.m_LoadInitF->isEnabled()==0) 
                    {	
                    //name1 = 0;path1 = 0;
                    m_Params[32] = 0;
                    m_Params[33] = 0;	
                    }

                    else {
                    m_Params[32] = name1;
                    m_Params[33] = path1;
                    }

                    std::cout<<"name1: "<<std::string(name1).c_str()<<"\n";
                    std::cout<<"path1: "<<std::string(path1).c_str()<<"\n";
                */
                std::cout<<"Host name: "<<dialog.HostName->text()<<"\n";

	



                std::cout<<"\n"<< m_Params[0];
                std::cout<<"\n"<< m_Params[1];
                std::cout<<"\n"<< m_Params[2];
                std::cout<<"\n"<< m_Params[4];
                std::cout<<"\n"<< m_Params[5];

                std::cout<<"\n"<< m_Params[6];
                std::cout<<"\n"<< m_Params[7];
                std::cout<<"\n"<<m_Params[8];
                std::cout<<"\n"<< m_Params[9];
                std::cout<<"\n"<<m_Params[10];
                std::cout<<"\n"<<m_Params[11];
                std::cout<<"\n"<<m_Params[12];
                std::cout<<"\n"<<m_Params[13];
                std::cout<<"\n"<<m_Params[14];
                std::cout<<"\n"<<std::string(m_Params[15]).c_str();
                std::cout<<"\n"<<std::string(m_Params[16]).c_str();
                std::cout<<"\n"<<m_Params[17]; // N is dimension( nx, ny)

                std::cout<<"\n"<<m_Params[18];
                std::cout<<"\n"<<m_Params[19];
                std::cout<<"\n"<<       m_Params[20];



                XmlRpcClient m_XmlRpcClient(dialog.HostName->text(), dialog.PortName->text().toInt());



                m_XmlRpcClient.execute("Reconstruct", m_Params, result);

                std::cout<<"\n End reconstruction on server \n";

              }


            else
              {

                switch(dialog.m_buttonGroup_2->selectedId())
                  {
                  case 0:
                    reconstruction->readFiles(name, path, N);
                    if(N%2 == 0 )  reconstruction->imageInterpolation();      //0928.
                    Object = reconstruction->Reconstruction3D(reconManner, iter, tau, eulers, m_Itercounts, phantom);

                    break;

                  case 1:
                    //Euler Angles.
                    eulers = (EulerAngles *)malloc(sizeof(EulerAngles));
                    eulers = phantomEulerAngles(rot, tilt, psi);

                    //reconstruction->SetJ12345Coeffs(reconj1, al, be, ga, la);
                    //reconstruction->Initialize(0, nx,ny,nz,m,alpha, fac, Volume);
                    //reconstruction->setTolNvRmatGdSize(tolnv);

                    Object = reconstruction->Reconstruction3D(reconManner, iter, tau, eulers, m_Itercounts, phantom);
                    free(eulers);
                    break;
                  }

              }

          }



        if(m_Itercounts > 1 )
          {

            if(dialog.m_buttonGroup->selectedId()==1)
              reconstruction->SetJ12345Coeffs(reconj1, al, be, ga, la);

            if(Default_newnv != newnv ) {reconstruction->SetNewNv(newnv);Default_newnv = newnv;}
            if(Default_bandwidth != bandwidth ) {reconstruction->SetBandWidth(bandwidth);Default_bandwidth = bandwidth;}
            if(Default_flow != flow ) {reconstruction->SetFlow(flow);Default_flow = flow;}
            reconstruction->setThick(thickness);
            //reconstruction->Phantoms(nx,ny,nz,nview,phantom);
            Object = reconstruction->Reconstruction3D(reconManner, iter, tau, eulers, m_Itercounts, phantom);

            if(dialog.m_RemoteReconRun->isChecked() == 1)

              {

                std::cout<<"\n Begin reconstruction on server \n";
	
                using namespace XmlRpc;	

                XmlRpc::XmlRpcValue m_Params, result;

	
                m_Params[0] = reconManner;
                m_Params[1] = iter;
                m_Params[2] = tau;
                //	m_Params[3] = eulers;		
                m_Params[4] = m_Itercounts;
                m_Params[5] = phantom;

                m_Params[6] = thickness;
                m_Params[7] = flow;
                m_Params[8] = bandwidth;
                m_Params[9] = newnv;
                m_Params[10] = reconj1;
                m_Params[11] = al;
                m_Params[12] = be;
                m_Params[13] = ga;
                m_Params[14] = la;

                m_Params[15] = name;
                m_Params[16] = path;
                m_Params[17] = N; // N is dimension( nx, ny)

                m_Params[18] = rot;
                m_Params[19] = tilt;
                m_Params[20] = psi;

                std::cout<<"Host name: "<<dialog.HostName->text()<<"\n";

                XmlRpcClient m_XmlRpcClient(dialog.HostName->text(), dialog.PortName->text().toInt());
	

                m_XmlRpcClient.execute("Reconstruct", m_Params, result);
	
                std::cout<<"\n End reconstruction on server \n";

              }	


          } 






        reconstruction->GlobalMeanError(Object);

        //Result = reconstruction->ConvertToVolume(Object);
        reconstruction->SaveVolume(Object);
        reconstruction->kill_all_but_main_img(Object);
        free(Object);








        /*

        try{
        VolMagick::Volume result_vol(Result.get<1>());
        QDir tmpdir(m_CacheDir.absPath() + "/tmp");
        if(!tmpdir.exists()) tmpdir.mkdir(tmpdir.absPath());
        QFileInfo tmpfile(tmpdir,"tmp.rawiv");
        qDebug("Creating volume %s",tmpfile.absFilePath().ascii());
        QString newfilename = QDir::convertSeparators(tmpfile.absFilePath());
        VolMagick::createVolumeFile(newfilename,
        result_vol.boundingBox(),
        result_vol.dimension(),
        std::vector<VolMagick::VoxelType>(1, result_vol.voxelType()));
        VolMagick::writeVolumeFile(result_vol,newfilename);
        openFile(newfilename);
        }



        catch(const VolMagick::Exception& e)
        {
        QMessageBox::critical(this,"Error loading Reconstruction volume",e.what());
        return false;
        }
        */



    }

  return true;


}

#endif
