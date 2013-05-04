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

/* $Id: newvolumemainwindow.h 3501 2011-01-25 18:20:56Z arand $ */

#include <boost/shared_array.hpp>
#include <boost/scoped_array.hpp>
#include "newvolumemainwindowbase.Qt3.h"
#include <VolumeRover/segmentationdialogimpl.h>
#include "pedetectiondialog.Qt3.h"
#include <VolumeFileTypes/RawIVSimpleSource.h>
//#include <qsocket.h>
#include <qdir.h>
#include <qdatetime.h>
#include <qthread.h>
//#include <qguardedptr.h>
//#include <qvbox.h>
#include <qimage.h>
//Added by qt3to4:
#include <QTimerEvent>
#include <QKeyEvent>
#include <QCustomEvent>

#include <QPointer>

#include <VolumeFileTypes/DataCutterSource.h>
#include <VolumeFileTypes/SourceManager.h>
#include <VolumeWidget/RenderableArray.h>
#include <VolumeRover/RoverRenderable.h>
#include <VolumeRover/ZoomedInVolume.h>
#include <VolumeRover/ZoomedOutVolume.h>
#include <Contouring/MultiContour.h>
//#include "DownLoadManager.h"
#include <VolumeRover/RecentFiles.h>
#include <Contouring/Contour.h>
#include <VolumeWidget/Extents.h>

#include <RenderServers/Corba.h>

#include <XmlRPC/XmlRpc.h>
#include <ColorTable/ColorTable.h>

#include <VolMagick/VolMagick.h>

#ifdef USING_MSLEVELSET
#include <MSLevelSet/levelset3D.h>
#include <FastContouring/FastContouring.h>
#endif

#ifdef USING_SECONDARYSTRUCTURES
#include <SecondaryStructureDataManager/SecondaryStructureData.h>
#endif

#ifdef USING_RECONSTRUCTION
#include <Reconstruction/Reconstruction.h>
#endif

#ifdef VOLUMEGRIDROVER
#include <VolumeGridRover/VolumeGridRover.h>
#endif

class Q3Process;
class Terminal;
//class ColorTable2D;
class Q3WidgetStack;
class MappedVolumeFile;

///\class NewVolumeMainWindow newvolumemainwindow.h
///\brief NewVolumeMainWindow is the ringleader of the entire dog and pony
///	show. It is the guts of the main window in Volume Rover and calls code from
///	nearly every other part of the program directly. Most of the functions are
///	slots which are called by the generated UI code. For more details about
///	the hooks between the UI and the program code, you need to look at 
///	newvolumemainwindowbase.ui in Qt Designer. Note that most of the
///	slots described here are not classified as slots. This is because they are 
///	virtual functions inherited from the base class NewVolumeMainWindowBase
///	where they _are_ slots.
///\author Anthony Thane
///\author John Wiggins
///\author Jose Rivera

///\enum NewVolumeMainWindow::UpdateMethod
///\brief These values are used to describe how the left (sub-volume) view
///	will be updated when the rover widget is manipulated.

///\var NewVolumeMainWindow::UpdateMethod NewVolumeMainWindow::UMInteractive
///	The sub-volume view updates as the rover widget is moved.

///\var NewVolumeMainWindow::UpdateMethod NewVolumeMainWindow::UMDelayed
///	The sub-volume view updates after the rover widget has stopped moving and
///	the mouse button is released.

///\var NewVolumeMainWindow::UpdateMethod NewVolumeMainWindow::UMManual
///	The sub-volume view is updated manually via a menu.

///\enum NewVolumeMainWindow::TransferFuncType
///\brief These values are used to describe the type (i.e. dimensionality) of the
///       transfer function.

///\var NewVolumeMainWindow::TransferFuncType NewVolumeMainWindow::TF1D
///     The standard 1-D transfer function.

///\var NewVolumeMainWindow::TransferFuncType NewVolumeMainWindow::TF2D
///     An experimental 2-D transfer function. (TODO: write more)

///\var NewVolumeMainWindow::TransferFuncType NewVolumeMainWindow::TF3D
///     An experimental 3-D transfer function. (TODO: write more)

class Animation;
class RenderServer;

class NewVolumeMainWindow : public NewVolumeMainWindowBase
{
	Q_OBJECT
public:
	enum UpdateMethod { UMInteractive, UMDelayed, UMManual };
	enum TransferFuncType { TF1D, TF2D, TF3D };

	NewVolumeMainWindow( QWidget* parent = 0, const char* name = 0, Qt::WFlags f = Qt::WType_TopLevel );
	virtual ~NewVolumeMainWindow();

	void init(); //call this after the main window has been shown.  This sets up OpenGL context dependent stuff...

///\fn void NewVolumeMainWindow::openFile(const QString& fileName)
///\brief This function is called from actionSlot() to load a volume file.
///\param fileName A QString containing the path to the file.
	void openFile(const QString& fileName);

///\fn void NewVolumeMainWindow::optionsSlot()
///\brief This slot brings up the options dialog and makes configuration
///	changes to the program after the window has been dismissed. Look at
///	optionsdialog.ui in Qt Designer for more details.
	void optionsSlot();
///\fn void NewVolumeMainWindow::actionSlot()
///\brief This slot launches a QFileDialog and opens the selected file when that
///	dialog is dismissed.
	void actionSlot();
///\fn void NewVolumeMainWindow::connectToDCSlot()
///\brief This slot initiates a DataCutter connection.
	void connectToDCSlot();
///\fn void NewVolumeMainWindow::functionChangedSlot()
///\brief This slot is triggered whenever the ColorTable widget (transfer
///	function) changes. Changes to isocontour nodes are not caught by this
/// slot, only changes to the color and alpha mapping.
	void functionChangedSlot();
///\fn void NewVolumeMainWindow::setExplorerQualitySlot(int value)
///\brief This slot is triggered whenever the "Explorer Render Quality" slider
///	is moved.
///\param value A value between 0 and 99 which represents the new quality.
	void setExplorerQualitySlot(int value);
///\fn void NewVolumeMainWindow::setMainQualitySlot(int value)
///\brief This slot is triggered whenever the "Main Render Quality" slider is
///	moved.
///\param value A value between 0 and 99 which represents the new quality.
	void setMainQualitySlot(int value);
///\fn void NewVolumeMainWindow::zoomedInClipSlot(int value)
///\brief This slot is triggered whenever the "Main Near Clip Plane" slider is
///	moved.
///\param value A value between 0 and 99 which represents the new clip plane
///	position.
	void zoomedInClipSlot(int value);
///\fn void NewVolumeMainWindow::zoomedOutClipSlot(int value)
///\brief This slot is triggered whenever the "Explorer Near Clip Plane"
///	slider is moved.
///\param value A value between 0 and 99 which represents the new clip plane
///	position.
	void zoomedOutClipSlot(int value);
///\fn void NewVolumeMainWindow::mouseReleasedMain()
///\brief This slot is triggered at the end of any mouse interaction in the
///	left ("Main") view. It copies the rotation information to the right
///	("Explorer") vie, keeping the two views in sync.
	void mouseReleasedMain();
///\fn void NewVolumeMainWindow::mouseReleasedPreview()
///\brief This slot is triggered at the end of any mouse interaction in the
///	right ("Explorer") view. It copies the rotation information to the left
///	("Main") view, keeping the two views in sync.
	void mouseReleasedPreview();
///\fn void NewVolumeMainWindow::centerSlot()
///\brief This slot is triggered by a menu item. It changes the camera look-at
///	point to the center of the Rover widget.
	void centerSlot();

///\fn void NewVolumeMainWindow::getThumbnail()
///\brief This function is called whenever volume data needs to be reloaded in
///	the right view.
	void getThumbnail();
///\fn void NewVolumeMainWindow::getContourSpectrum()
///\brief This function reads the contour spectrum for the current variable
///	and time step from the currently opened file's cache and passes it to the 
/// ColorTable widget.
	void getContourSpectrum();
///\fn void NewVolumeMainWindow::getContourTree()
///\brief This function reads the contour tree for the current variable and 
///	time step from the currently opened file's cache and passes it to the 
/// ColorTable widget.
	void getContourTree();

///\fn void NewVolumeMainWindow::acquireConSpecSlot()
///\brief This slot is triggered by the ColorTable widget. It computes the
///	contour spectrum for the current variable and time step and then calls
///	getContourSpectrum().
	void acquireConSpecSlot();
///\fn void NewVolumeMainWindow::acquireConTreeSlot()
///\brief This slot is triggered by the ColorTable widget. It computes the
///	contour tree for the current variable and time step and then calls
///	getContourTree().
	void acquireConTreeSlot();

///\fn void NewVolumeMainWindow::explorerChangedSlot()
///\brief This slot is triggered whenever the volume data in the left view
///	needs to be reloaded. This can happen in response to a manual update request
///	by the user. This function is also called from several other functions.
	void explorerChangedSlot();
///\fn void NewVolumeMainWindow::explorerMoveSlot()
///\brief This slot is triggered whenever the Rover widget is moved.
	void explorerMoveSlot();
///\fn void NewVolumeMainWindow::explorerReleaseSlot()
///\brief This slot is triggered whenever the Rover widget has been moved and
///	the mouse button is released.
	void explorerReleaseSlot();

///\fn void NewVolumeMainWindow::variableOrTimeChangeSlot()
///\brief This slot is triggered whenever the current variable or time step
///	changes.
	void variableOrTimeChangeSlot();

///\fn void NewVolumeMainWindow::toggleWireframeRenderingSlot(bool state)
///\brief This slot is triggered by a menu item. It enables or disables
///	wireframe rendering of isosurfaces and loaded geometry.
///\param state true -> enables wireframes, false -> disables wireframes
	void toggleWireframeRenderingSlot(bool state);

	void toggleRenderSurfaceWithWireframeSlot(bool state);
	
///\fn void NewVolumeMainWindow::toggleWireCubeSlot(bool state)
///\brief This slot is triggered by a menu item. It enables or disables
///	drawing of the Rover widget and bounding box wireframes.
///\param state true -> enables drawing, false -> disables drawing
	void toggleWireCubeSlot(bool state);
///\fn void NewVolumeMainWindow::toggleDepthCueSlot(bool state)
///\brief This slot is triggered by a menu item. It enables or disables depth
///	cueing in both views.
///\param state true -> enables depth cueing, false -> disables depth cueing
	void toggleDepthCueSlot(bool state);

///\fn void NewVolumeMainWindow::isocontourNodeChangedSlot(int node, double value)
///\brief This slot is triggered by the ColorTable widget. This happens
///	whenever an isocontour node is moved in the ColorTable. It notifies the
///	MultiContour objects of the changes and redraws the views.
///\param node This is the ID of the isocontour that changed
///\param value This is the new isovalue for the node. It's in the range
///	[0,1].
	void isocontourNodeChangedSlot(int node, double value);
	void isocontourAskIsovalueSlot(int node);
///\fn void NewVolumeMainWindow::isocontourNodeColorChangedSlot(int node, double R, double G, double B)
///\brief This slot is triggered by the ColorTable widget. This happens
///	whenever the color of an isocontour node is edited. It notifies the
///	MultiContour objects of the changes and redraws the views.
///\param node This is the ID of the isocontour that changed.
///\param R This is the red component of the new color. Range [0,1].
///\param G This is the green component of the new color. Range [0,1].
///\param B This is the blue component of the new color. Range [0,1].
	void isocontourNodeColorChangedSlot(int node, double R, double G, double B);
///\fn void NewVolumeMainWindow::isocontourNodeAddedSlot(int node, double value, double R, double G, double B)
///\brief This slot is triggered by the ColorTable widget. This happens
///	whenever an isocontour node is added. It notifies the MultiContour objects
/// of the changes and redraws the views.
///\param node This is the ID of the new isocontour.
///\param value This is the isovalue of the new isocontour. Range [0,1].
///\param R This is the red component of the isocontour's color. Range [0,1].
///\param G This is the green component of the isocontour's color. Range [0,1].
///\param B This is the blue component of the isocontour's color. Range [0,1].
	void isocontourNodeAddedSlot(int node, double value, double R, double G, double B);
///\fn void NewVolumeMainWindow::isocontourNodeDeletedSlot(int node)
///\brief This slot is triggered by the ColorTable widget. This happens
///	whenever an isocontour node is removed. It notifies the MultiContour objects
/// of the changes and redraws the views.
///\param node This is the ID of the new isocontour.
	void isocontourNodeDeletedSlot(int node);
///\fn void NewVolumeMainWindow::isocontourNodesAllChangedSlot()
///\brief This slot is triggered by the ColorTable widget. This happens
///	whenever a .vinay file is loaded. It removes any contours from the
///	MultiContour objects and adds (or doesn't) contours from the new transfer
///	function, then redraws the views.
	void isocontourNodesAllChangedSlot();
	
///\fn void NewVolumeMainWindow::contourTreeNodeChangedSlot(int node, double value)
///\brief This slot is triggered by the ColorTable widget. This happens
///	whenever a contour tree node is moved. Currently, no further action is
///	taken.
///\param node This is the ID of the isocontour associated with this node.
///\param value This is the new isovalue for this node.
	void contourTreeNodeChangedSlot(int node, double value);
	//void contourTreeNodeColorChangedSlot(int node, double R, double G, double B);
///\fn void NewVolumeMainWindow::contourTreeNodeAddedSlot(int node, int edge, double value)
///\brief This slot is triggered by the ColorTable widget. This happens
///	whenever a contour tree node is added. Currently, no further action is
///	taken.
///\param node This is the ID of the isocontour associated with this node.
///\param edge This is the index of edge that this node is on in the contour tree.
///\param value This is the isovalue for this node.
	void contourTreeNodeAddedSlot(int node, int edge, double value);
///\fn void NewVolumeMainWindow::contourTreeNodeDeletedSlot(int node)
///\brief This slot is triggered by the ColorTable widget. This happens
///	whenever a contour tree node is removed. Currently, no further action is
///	taken.
///\param node This is the ID of the isocontour associated with this node.
	void contourTreeNodeDeletedSlot(int node);
///\fn void NewVolumeMainWindow::contourTreeNodesAllChangedSlot()
///\brief Not currently implemented. (There isn't even a definition in
///	newvolumemainwindow.cpp)
	void contourTreeNodesAllChangedSlot();

///\fn void NewVolumeMainWindow::connectServerSlot()
///\brief This slot is triggered by a menu item. It initiates a connection to
///	a render server after showing a ServerSelectorDialog.
	void connectServerSlot();
///\fn void NewVolumeMainWindow::disconnectServerSlot()
///\brief This slot is triggered by a menu item. It closes any connection
///	established with a render server.
	void disconnectServerSlot();
///\fn void NewVolumeMainWindow::serverSettingsSlot()
///\brief This slot is triggered by a menu item. It shows a settings dialog
///	which is appropriate for the render server that has been connected to. After
///	the dialog is dismissed, the new settings are sent to the server.
	void serverSettingsSlot();
///\fn void NewVolumeMainWindow::renderFrameSlot()
///\brief This slot is triggered by a menu item. It sends a render command to
///	the render server that has been connected to.
	void renderFrameSlot();
///\fn void NewVolumeMainWindow::renderAnimationSlot()
///\brief Does nothing.
	void renderAnimationSlot();

///\fn void NewVolumeMainWindow::bilateralFilterSlot()
///\brief This slot is triggered by a menu item. It initiates the bilateral
///	filtering process on the volume data loaded into the left view.
	void bilateralFilterSlot();
///\fn void NewVolumeMainWindow::virusSegmentationSlot()
///\brief This slot is triggered by a menu item. It shows a SegmentationDialog
///	and then possibly initiates the virus segmentation code on the data in the
///	left view.
	void virusSegmentationSlot();

	void contrastEnhancementSlot();

	void PEDetectionSlot();

	void pocketTunnelSlot();

	void smoothGeometrySlot();

	void anisotropicDiffusionSlot();

	void sliceRenderingSlot();

	void boundaryPointCloudSlot();

	void tightCoconeSlot();

	void curationSlot();

	void skeletonizationSlot();

	void secondaryStructureElucidationSlot();

	void clipGeometryToVolumeBoxSlot(bool);

	void saveSkeletonSlot();

	void clearSkeletonSlot();

	void signedDistanceFunctionSlot();

	void mergeGeometrySlot();

	void convertIsosurfaceToGeometrySlot();

        void highLevelSetReconSlot();       

	void highLevelSetSlot();

	void LBIEMeshingSlot();
	
	void LBIEQualityImprovementSlot();

	void openImageFileSlot();

	void projectGeometrySlot();

	void gdtvFilterSlot();

	void colorGeometryByVolumeSlot();

	void cullGeometryWithSubvolumeBoxSlot();

	void syncIsocontourValuesWithVolumeGridRover();

        bool ReconstructionFromProjectionSlot();
	
///\fn void NewVolumeMainWindow::MSLevelSetSlot()
///\brief This slot is triggered bya menu item. It invokes the  Modified Mumford-Shah level set segmentation.
	void MSLevelSetSlot();

///\fn void NewVolumeMainWindow::saveSubvolumeSlot()
///\brief This slot is triggered by a menu item. It uses a VolumeTranscriber
///	to save the volume in the left view to a new file.
	void saveSubvolumeSlot();
///\fn void NewVolumeMainWindow::saveImageSlot()
///\brief This slot is triggered by a menu item. It shows an ImageSaveDialog
///	and then possibly writes the left or right view to an image file using the
///	Qt library.
	void saveImageSlot();

///\fn void NewVolumeMainWindow::startRecordingAnimationSlot()
///\brief This slot is triggered by a menu item. It constructs an Animation
///	instance and starts a timer.
	void startRecordingAnimationSlot();
///\fn void NewVolumeMainWindow::stopRecordingAnimationSlot()
///\brief This slot is triggered by a menu item. It stops the timer started in
///	startRecordingAnimationSlot() but leaves the Animation object untouched.
	void stopRecordingAnimationSlot();
///\fn void NewVolumeMainWindow::playAnimationSlot()
///\brief This slot is triggered by a menu item. It checks to make sure that
///	an Animation has been created or loaded and that recording is not currently
///	enabled. If these conditions are met, a timer is started and some flags are
///	set.
	void playAnimationSlot();
///\fn void NewVolumeMainWindow::stopAnimationSlot()
///\brief This slot is triggered by a menu item. It stops the timer started in
///	playAnimationSlot(), clears some flags and then resets the camera in the
///	right ("Explorer") view.
	void stopAnimationSlot();
///\fn void NewVolumeMainWindow::saveAnimationSlot()
///\brief This slot is triggered by a menu item. It checks to see if there is
///	an Animation instance and recording is not taking place. If these conditions
///	are met, the animation is written to a file that has been specified by the
///	user.
	void saveAnimationSlot();
///\fn void NewVolumeMainWindow::loadAnimationSlot()
///\brief This slot is triggered by a menu item. It performs some sanity
///	checks and then loads a previously saved animation from a user specified
///	file.
	void loadAnimationSlot();
///\fn void NewVolumeMainWindow::renderSequenceSlot()
///\brief This slot is triggered by a menu item. It prompts the user for a
///	filename and this calls playAnimationSlot(). After that it makes sure that
///	the animation is playing before setting an extra flag.
	void renderSequenceSlot();

///\fn unsigned int NewVolumeMainWindow::getVarNum() const
///\brief This function returns the current variable index.
///\return The current variable index. 
	unsigned int getVarNum() const;
///\fn unsigned int NewVolumeMainWindow::getTimeStep() const
///\brief This function returns the current time step.
///\return The current time step.
	unsigned int getTimeStep() const;

///\fn void NewVolumeMainWindow::loadGeometrySlot()
///\brief This slot is triggered by a menu item. It prompts the user to select
///	a geometry file and then adds the loaded geometry to the RenderableArray
///	m_Geometries.
	void loadGeometrySlot();
///\fn void NewVolumeMainWindow::clearGeometrySlot()
///\brief This slot is triggered by a menu item. It clears the RenderableArray
///	m_Geometries.
	void clearGeometrySlot();
///\fn void NewVolumeMainWindow::saveGeometrySlot()
///\brief This slot is triggered by a menu item.  It saves the RenderableArray m_Geometries.
	void saveGeometrySlot();

///\fn void NewVolumeMainWindow::exportZoomedInIsosurfaceSlot()
///\brief This item is triggered by a menu item. It prompts the user for a
///	filename and then saves the isosurface(s) in the left view to a new file.
	void exportZoomedInIsosurfaceSlot();
///\fn void NewVolumeMainWindow::exportZoomedOutIsosurfaceSlot()
///\brief This item is triggered by a menu item. It prompts the user for a
///	filename and then saves the isosurface(s) in the right view to a new file.
	void exportZoomedOutIsosurfaceSlot();

///\fn void NewVolumeMainWindow::resetGeometryTransformationSlot()
///\brief This item is triggered by a menu item. This resets any
//transformations that have accumulated in the GeometryRenderer instance
//controled by m_ThumbnailRenderable.
	void resetGeometryTransformationSlot();
///\fn void NewVolumeMainWindow::toggleGeometryTransformationSlot()
///\brief This item is triggered by a menu item. This switches mouse control
///	between controlling the camera and controling the orientation/position of
///	loaded geometry. Changes only affect the right view.
	void toggleGeometryTransformationSlot();
///\fn void NewVolumeMainWindow::toggleTerminalSlot()
///\brief This slot is triggered by a menu item.  When called, it either
///       shows or hides the Volume Rover terminal window.
	void toggleTerminalSlot(bool show);

///\fn Geometry* NewVolumeMainWindow::loadGeometry(const char* filename) const
///\brief This is an old function whose purpose has been reassigned to
///	loadGeometrySlot().
	Geometry* loadGeometry(const char* filename) const;
///\fn bool NewVolumeMainWindow::saveGeometry(const char* filename, Geometry* geometry) const
///\brief As with loadGeometry, this function has is superceded by the
///	exportZoomed*IsosurfaceSlot() functions.
	bool saveGeometry(const char* filename, Geometry* geometry) const;

///\fn QColor NewVolumeMainWindow::getSavedColor()
///\brief This function retrieves the saved background color from Volume
///	Rover's settings file.
///\return A QColor object containing the background color or black if no
///	color setting is found.
	QColor getSavedColor();
///\fn void NewVolumeMainWindow::setSavedColor(const QColor& color)
///\brief This function saves the specified color to Volume Rover's settings
///	file.
///\param color A QColor containing the background color.
	void setSavedColor(const QColor& color);
///\fn QDir NewVolumeMainWindow::getCacheDir()
///\brief This function retrieves the path to the cache directory from Volume
///	Rover's settings file. The user will be prompted to select a directory if
///	needed.
///\return A QDir instance with a valid path.
	QDir getCacheDir();
///\fn QDir NewVolumeMainWindow::presentDirDialog(QDir defaultDir)
///\brief This is a convenience function called by getCacheDir() to do its
///	bidding.
///\return A QDir instance with a valid path.
	QDir presentDirDialog(QDir defaultDir);

///\fn Terminal* NewVolumeMainWindow::getTerminal() const
///\brief Returns the volume rover terminal.
///\return A Terminal widget.
	Terminal *getTerminal() const;

public slots:
///\fn void NewVolumeMainWindow::recentFileSlot(int fileNum)
///\brief This slot is not currently in use.
	void recentFileSlot(int fileNum);
///\fn void NewVolumeMainWindow::finishConnectingToDCSlot()
///\brief This slot is triggered by the QSocket constructed in
///	connectToDCSlot(). It finishes the task of connecting to a DataCutter
///	server.
	void finishConnectingToDCSlot();
///\fn void NewVolumeMainWindow::errorConnectingToDCSlot(int num)
///\brief This is the error handler for the QSocket created in
///	connectToDCSlot().
	void errorConnectingToDCSlot(int num);

///\fn void NewVolumeMainWindow::openFileSlot(const QString& filename)
///\brief Simply calls openFile()
	void openFileSlot(const QString& filename);

///\fn void NewVolumeMainWindow::toggleVolumeGridRoverSlot()
///\brief This slot is triggered by a menu item.  When called, it either
///       shows or hides the Volume Grid Rover window.
//	void toggleVolumeGridRoverSlot(bool show);

protected slots:

#ifdef VOLUMEGRIDROVER
	void gridRoverDepthChangedSlot(SliceCanvas *sc, int d);
#endif

	void receiveTilingGeometrySlot(const boost::shared_ptr<Geometry>& g);
	void showImage(const QImage& img);

protected:
///\fn virtual void NewVolumeMainWindow::keyPressEvent(QKeyEvent *e)
///\brief This is a QWidget keyboard event callback. It looks for shift key
///	presses so that it can swap out some MouseHandler instances. This is a small
///	hack for Dr. Bajaj's laptop. Basically, it emulates a right mouse button
///	click using the left mouse button.
	virtual void keyPressEvent(QKeyEvent *e);
///\fn virtual void NewVolumeMainWindow::keyReleaseEvent(QKeyEvent *e)
///\brief See keyPressEvent().
	virtual void keyReleaseEvent(QKeyEvent *e);
///\fn virtual void NewVolumeMainWindow::timerEvent(QTimerEvent *e)
///\brief This QWidget callback catches timer events for animation purposes.
///	It handles both recording and playback based on various boolean flags.
	virtual void timerEvent(QTimerEvent *e);

///\fn inline unsigned int NewVolumeMainWindow::upToPowerOfTwo(unsigned int value) const
///\brief This function generates a power of two from some number.
///\param value A positive number.
///\return A power of two which is greater than or equal to the passed in
///	value.
	inline unsigned int upToPowerOfTwo(unsigned int value) const;
	///\fn inline double NewVolumeMainWindow::texCoordOfSample(double sample, int bufferWidth, int canvasWidth, double bufferMin, double bufferMax) const
///\brief This function converts object space coordinates to texture space
///	coordinates.
	inline double texCoordOfSample(double sample, int bufferWidth, int canvasWidth, double bufferMin, double bufferMax) const;
///\fn void NewVolumeMainWindow::copyToUploadableBufferDensity(RoverRenderable* roverRenderable, Extents* extents, unsigned int var)
///\brief This function copies data from a VolumeBuffer that may not have
///	power-of-two dimensions to an upload buffer which does have power-of-two
///	dimensions.
///\param roverRenderable The RoverRenderable that will be uploaded to.
///\param extents Not used.
///\param var The index of the variable to be copied.
	void copyToUploadableBufferDensity(RoverRenderable* roverRenderable, Extents* extents, unsigned int var);
///\fn void NewVolumeMainWindow::copyToUploadableBufferGradient(RoverRenderable* roverRenderable, Extents* extents, unsigned int var)
///\brief This function copies gradient data from a VolumeBuffer that may not
///	have power-of-two dimensions to an upload buffer which does have
///	power-of-two dimensions.
///\param roverRenderable The RoverRenderable that will be uploaded to.
///\param extents Not used.
///\param var The index of the variable to be copied.
	void copyToUploadableBufferGradient(RoverRenderable* roverRenderable, Extents* extents, unsigned int var);
///\fn void NewVolumeMainWindow::copyToUploadableBufferRGBA(RoverRenderable* roverRenderable, Extents* extents, unsigned int var, unsigned int offset)
///\brief This function copies one component of an RGBA volume from a
///	VolumeBuffer that may not have power-of-two dimensions to part of an upload
///	buffer which does have power-of-two dimensions.
///\param roverRenderable The RoverRenderable that will be uploaded to.
///\param extents Not used.
///\param var The index of the variable to be copied.
///\param offset 0,1,2, or 3 depending on whether the red,green,blue, or alpha variable is being copied.
	void copyToUploadableBufferRGBA(RoverRenderable* roverRenderable, Extents* extents, unsigned int var, unsigned int offset);

///\fn unsigned int NewVolumeMainWindow::getRedVariable() const
///\brief This function returns the index of the current red variable. This
///	only gives valid results during RGBA rendering.
///\return A number specifying the index of the red variable.
	unsigned int getRedVariable() const;
///\fn unsigned int NewVolumeMainWindow::getGreenVariable() const
///\brief This function returns the index of the current green variable. This
///	only gives valid results during RGBA rendering.
///\return A number specifying the index of the green variable.
	unsigned int getGreenVariable() const;
///\fn unsigned int NewVolumeMainWindow::getBlueVariable() const
///\brief This function returns the index of the current blue variable. This
///	only gives valid results during RGBA rendering.
///\return A number specifying the index of the blue variable.
	unsigned int getBlueVariable() const;
///\fn unsigned int NewVolumeMainWindow::getAlphaVariable() const
///\brief This function returns the index of the current alpha variable. This
///	only gives valid results during RGBA rendering (kinda sorta).
///\return A number specifying the index of the alpha variable.
	unsigned int getAlphaVariable() const;

///\fn void NewVolumeMainWindow::updateRoverRenderable(RoverRenderable* roverRenderable, Extents* extents)
///\brief This function uploads volume data to the specified RoverRenderable
///	instance.
///\param roverRenderable The RoverRenderable instance to be updated.
///\param extents An Extents object describing the bounding cube of the volume.
	void updateRoverRenderable(RoverRenderable* roverRenderable, Extents* extents);

///\fn void NewVolumeMainWindow::updateRecentlyUsedList(const QString& filename)
///\brief This function adds a filename to the recently opened list.
///\warning This function is currently not being used.
///\param filename A QString specifying a valid path to a file.
	void updateRecentlyUsedList(const QString& filename);
///\fn void NewVolumeMainWindow::checkForConnection()
///\brief This function toggles the availability of certain menu items based
///	on the presence or lack of a connection to a RenderServer.
	void checkForConnection();
///\fn bool NewVolumeMainWindow::checkError()
///\brief This function checks to see if any errors have occured during
///	acquisition of volume data.
	bool checkError();
///\fn void NewVolumeMainWindow::setUpdateMethod(VolumeSource::DownLoadFrequency method)
///\brief This function sets up the behavior of updates when the Rover widget
///	is manipulated. It basically maps a VolumeSource::DownLoadFrequency to a
///	NewVolumeMainWindow::UpdateMethod.
	void setUpdateMethod(VolumeSource::DownLoadFrequency method);
///\fn void NewVolumeMainWindow::updateVariableInfo( bool flush )
///\brief This function updates various UI elements in response to a new file
///	being loaded or the rendering mode switching between colormapped and RGBA
///	rendering.
///\param flush If this parameter is true, some extra variables are cleared. This option is used when loading a new file.
	void updateVariableInfo( bool flush );

///\fn void NewVolumeMainWindow::enableDensityToolbar()
///\brief This function enables the variable selection UI associated with
///	colormapped rendering.
	void enableDensityToolbar();
///\fn void NewVolumeMainWindow::enableRGBAToolbar()
///\brief This function enables the variable selection UI associated with RGBA
///	rendering.
	void enableRGBAToolbar();
///\fn void NewVolumeMainWindow::disableDensityToolbar()
///\brief This function disables the variable selection UI associated with
///	colormapped rendering.
	void disableDensityToolbar();
///\fn void NewVolumeMainWindow::disableRGBAToolbar()
///\brief This function disables the variable selection UI associated with
///	RGBA rendering.
	void disableRGBAToolbar();
///\fn void NewVolumeMainWindow::enableVariableBox(QComboBox* box, unsigned int var)
///\brief This function resets a QComboBox and populates it with variable
///	names.
///\param box The QComboBox to modify.
///\param var The variable index to leave selected after box has been populated.
	void enableVariableBox(QComboBox* box, unsigned int var);

///\fn void NewVolumeMainWindow::recordFrame()
///\brief This function adds the current viewing state for the right view to
///	the Animation instance m_Animation. This is where keyframes are created when
///	recording an animation.
	void recordFrame();
///\fn void NewVolumeMainWindow::customEvent(QCustomEvent *ev)
///\brief This function is not currently used.
	void customEvent(QCustomEvent *ev);

	//DownLoadManager m_DownLoadManager;
	SourceManager m_SourceManager;
	QString m_CurrentVolumeFilename;
	

	RenderableArray m_Geometries;
#ifdef USING_SECONDARYSTRUCTURES
	SecondaryStructureData *ssData;
#endif
	RecentFiles m_RecentFiles;

	ZoomedInVolume m_ZoomedInRenderable;
	Extents m_ZoomedInExtents;
	ZoomedOutVolume m_ThumbnailRenderable;
	Extents m_ThumbnailExtents;

	//unsigned char* m_UploadBuffer;
	boost::shared_array<unsigned char> m_UploadBuffer;

	//VolumeSource* m_VolumeSource;


	Q3Socket* m_PendingSocket;
	QDir m_CacheDir;
	UpdateMethod m_UpdateMethod;
	RenderServer* m_RenderServer;
	
	TransferFuncType m_TransferFunc;

	MouseHandler* m_SavedZoomedInHandler;
	MouseHandler* m_SavedZoomedOutHandler;

	Animation* m_Animation;
	QTime m_Time;
	QString m_AnimationFrameName;
	int m_AnimationTimerId;
	unsigned int m_FrameNumber;

#ifdef USING_RECONSTRUCTION
        int m_Itercounts;      //for accumulated iters step in Reconstrucion.
        int Default_newnv, Default_bandwidth, Default_flow, Default_thickness;
        int reconManner;
	Reconstruction *reconstruction ;
#endif

	int m_SavedDensityVar;
	int m_SavedTimeStep;

	QPointer<Terminal> m_Terminal;

#ifdef VOLUMEGRIDROVER
	VolumeGridRover *m_VolumeGridRover;
#endif

	Q3ToolBar *m_ColorToolbar;
	Q3WidgetStack *m_ColorToolbarStack;
	CVC::ColorTable *m_ColorTable;
	//	ColorTable2D *m_ColorTable2D;

	bool m_RGBARendering;
	bool m_WireFrame;
	bool m_AnimationRecording;
	bool m_AnimationPlaying;
	bool m_SaveAnimationFrame;
	bool m_TransformGeometry;
	bool m_SubVolumeIsFiltered; //Now ignored.  We are now giving the user a choice to do the full filtering immediately

	//MappedVolumeFile *m_MappedVolumeFile; /* mapped volume file for the volume grid rover */
	VolMagick::VolumeFileInfo m_VolumeFileInfo; /* volume file info for the volume grid rover */

	//ConDataset* m_ConDataset;
	//ContourManager m_ContourManager;
	class RemoteSegThread : public QThread 
	{
	public:
		RemoteSegThread(SegmentationDialog *dialog, NewVolumeMainWindow *nvmw, unsigned int stackSize = 0);
		virtual void run();
	private:
		NewVolumeMainWindow *m_NewVolumeMainWindow;
		XmlRpc::XmlRpcValue m_Params;
		int type;
		XmlRpc::XmlRpcClient m_XmlRpcClient;
	} *m_RemoteSegThread;
	
	class LocalSegThread : public QThread 
	{
    public:
		LocalSegThread(const char *filename, SegmentationDialog *dialog, NewVolumeMainWindow *nvmw, unsigned int stackSize = 0);
		virtual void run();
    private:
		NewVolumeMainWindow *m_NewVolumeMainWindow;
		XmlRpc::XmlRpcValue m_Params;
		int type;
	} *m_LocalSegThread;
	
#ifdef USING_PE_DETECTION
	class PEDetectionThread : public QThread
	{
	public:
	  PEDetectionThread(const QString& filename,
			    PEDetectionDialog *dialog, 
			    NewVolumeMainWindow *nvmw, 
			    unsigned int stackSize = 0);
	  virtual void run();

	private:
	  NewVolumeMainWindow *m_NewVolumeMainWindow;
	  XmlRpc::XmlRpcValue m_Params;
	  XmlRpc::XmlRpcClient m_XmlRpcClient;
	  QString m_Filename;
	  bool m_Remote;
	} *m_PEDetectionThread;
#endif

#ifdef USING_POCKET_TUNNEL
	class PocketTunnelThread : public QThread
	{
	public:
	  PocketTunnelThread(Geometry *inputMesh, 
			     NewVolumeMainWindow *nvmw,
			     unsigned int stackSize = 0);
	  virtual void run();
	private:
	  NewVolumeMainWindow *m_NewVolumeMainWindow;
	  Geometry *m_InputMesh;
	} *m_PocketTunnelThread;
#endif

#ifdef USING_MSLEVELSET
	class MSLevelSetThread : public QThread
	{
	public:
	  MSLevelSetThread(const MSLevelSetParams& params, 
			   const VolMagick::Volume& vol,
			   unsigned int stackSize = 0);
	  virtual void run();

	  static void sendUpdate(const float* vol,
				 int dimx,
				 int dimy,
				 int dimz,
				 void *context);
	  
	private:
	  MSLevelSetParams m_Params;
	  VolMagick::Volume m_VolData;
	  FastContouring::ContourExtractor m_ContourExtractor;
	} *m_MSLevelSetThread;
#endif

	//QVBox *m_ProgressBars;
	friend class MSLevelSetThread;
};
