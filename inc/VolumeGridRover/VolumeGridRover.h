/*
  Copyright 2005-2008 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeGridRover.

  VolumeGridRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeGridRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: VolumeGridRover.h 4741 2011-10-21 21:22:06Z transfix $ */

#ifndef VOLUMEGRIDROVER_H
#define VOLUMEGRIDROVER_H

#include <QList>
#include <QListView>
#include <QTransform>
#include <QPainter>
#include <QFrame>
#include <QPixmap>
#include <QSlider>
#include <QLineEdit>
#include <QComboBox>
#include <QSpinBox>
#include <QThread>
#include <QRect>
#include <QTabWidget>
#include <QImage>
#include <GL/glew.h>
#include <QGLViewer/qglviewer.h>
#include <QGLViewer/camera.h>

#include <VolMagick/VolMagick.h>

#include <VolumeGridRover/PointClassFile.h>
#include <VolumeGridRover/ContourFile.h>

#include <vector>
#include <set>
#include <list>
#include <boost/tuple/tuple.hpp>
#include <boost/multi_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/scoped_ptr.hpp>

#ifdef USING_EM_CLUSTERING
#include <VolumeGridRover/EM.h>
#endif

#include <VolumeGridRover/SurfRecon.h>
#include <cvcraw_geometry/Geometry.h>
#include <VolumeGridRover/SDF2D.h>

#ifdef USING_TILING
#include <Tiling/tiling.h>
#endif

enum CellMarkingMode { PointClassMarking, ContourMarking };

namespace Ui
{
  class VolumeGridRoverBase;
}

class GridPoint
{
 public:
  GridPoint(unsigned int _x = 0, unsigned int _y = 0, unsigned int _z = 0)
    { x = _x; y = _y; z = _z; }
  ~GridPoint(){}
  unsigned int x;
  unsigned int y; 
  unsigned int z;
};

/* 
   contains a set of points to define a class of points
   on the volume.
*/
class PointClass
{
 public:
  enum Plane { XY, XZ, ZY, ANY };
 
  PointClass(const QColor &color, const QString& name);
  ~PointClass();

  inline void setColor(QColor &color) { m_PointClassColor = color; }
  inline QColor& getColor() { return m_PointClassColor; }
  inline void setName(const QString &name) { m_Name = name; }
  inline QString& getName() { return m_Name; }
  inline void setRadius(int radius) { m_Radius = radius; }
  inline int getRadius() { return m_Radius; }
  
  inline QList<GridPoint*> &getPointList() { return m_PointList; }

  void addPoint(unsigned int x, unsigned int y, unsigned int z);
  bool removePoint(unsigned int x, unsigned int y, unsigned int z); /* returns true if remove is successful */

  bool pointIsInClass(unsigned int x, unsigned int y, unsigned int z);
  GridPoint *getClosestPoint(unsigned int x, unsigned int y, unsigned int z, Plane onPlane = ANY); /* return the closest point on the specified plane */

 private:
  QColor m_PointClassColor; /* the color associated with this point class */
  QString m_Name; /* the name of this point class */
  int m_Radius; /* the radius of each point as they are drawn */
  QList<GridPoint*> m_PointList; /* the list of all the points in this class */
};

class SliceCanvas;

class SliceRenderer
{
 public:
  SliceRenderer(SliceCanvas *canvas) : m_SliceCanvas(canvas) {}
  virtual ~SliceRenderer() {}

  virtual bool init() = 0;
  virtual void draw() = 0;
  virtual void uploadSlice(unsigned char *, unsigned int, unsigned int) = 0;
  virtual void uploadColorTable() = 0;

  SliceCanvas *m_SliceCanvas;
};

class ARBFragmentProgramSliceRenderer;
class PalettedSliceRenderer;
class SGIColorTableSliceRenderer;

/* handles drawing of volume slices */
class SliceCanvas : public QGLViewer
{
  Q_OBJECT

    public:
  enum SliceAxis { XY, XZ, ZY };
  SliceCanvas(SliceAxis a, QWidget * parent = 0, const char * name = 0);
  // TODO: Qt6 migration - QGLFormat removed, use QSurfaceFormat if needed
  // SliceCanvas(SliceAxis a, const QGLFormat& format, QWidget * parent = 0, const char * name = 0);
  ~SliceCanvas();

  void setVolume(const VolMagick::VolumeFileInfo& vfi);
  void unsetVolume();
  
  inline unsigned int getDepth() { return m_Depth; }

  inline SliceAxis getSliceAxis() { return m_SliceAxis; }
  inline void setSliceAxis(SliceAxis a) { m_SliceAxis = a; }

  void setTransferFunction(unsigned char*);
  const unsigned char* getTransferFunction() const { return m_ByteMap; }
  
  void setPointClassList(QList<PointClass*> ***list) { m_PointClassList = list; /*qDebug("setPointClassList(): %p",&m_PointClassList);*/ }
  //QPtrList<PointClass> *getPointClassList() { return m_PointClassList; }
  PointClass *getCurrentClass() { return m_CurrentPointClass; }

  void setContours(const SurfRecon::ContourPtrArray& contours) { m_Contours = contours; }
  void setSDF(const SDF2D::ImagePtr& sdf) { m_SDF = sdf; }

  void addPoint(unsigned int x, unsigned int y, unsigned int z);
  void removePoint(unsigned int x, unsigned int y, unsigned int z);

  bool isValid() { return m_Drawable; }

  VolMagick::Volume sliceData() const { return m_VolumeSlice; }

  public slots:
  void setDepth(int d);
  void setCurrentClass(int index);
  void setPointSize(int r);
  void setGreyScale(bool set);
  void setRenderSDF(bool set);
  void setRenderControlPoints(bool set);
  void setCurrentVariable(int var);
  void setCurrentTimestep(int time);
  void resetView(); /* centers the camera on the slice */
  void setCellMarkingMode(int m);
  void setCurrentContour(const std::string& name);

  //void setInterpolationType(int interp);
  
 protected:
  //void paintEvent(QPaintEvent *event);
  void mouseMoveEvent(QMouseEvent *event);
  void mouseDoubleClickEvent(QMouseEvent *event);
  void mousePressEvent(QMouseEvent *event);
  void mouseReleaseEvent(QMouseEvent *event);
  void keyPressEvent(QKeyEvent *event);

  /* rendering functions */
  virtual void init();
  virtual void draw();
  //dimx and dimy must be 2^n
  virtual void uploadSlice(unsigned char *slice, unsigned int dimx, unsigned int dimy);
  virtual void uploadColorTable();

  void drawPointClasses();
  void drawContours(bool withNames = false);
  void drawSelectionRectangle();

  //selection functions
  virtual void drawWithNames(); //draw the control points for selection
  virtual void endSelection(const QPoint&);
  void addIdToSelection(int id);
  void removeIdFromSelection(int id);

  virtual void postDraw(); //overloaded so we can draw the non slice elements here (points, contours, selection rectangle)

  void deleteSelected();

  //get object space coordinates from screen coordinates
  qglviewer::Vec getObjectSpaceCoords(const qglviewer::Vec& screenCoords) const;

  void updateSlice();

 signals:
  void mouseOver(int x, int y, int z,
		 float objx, float objy, float objz,
		 int r, int g, int b, int a,
		 double value);
  void pointAdded(unsigned int var, unsigned int time, int classIndex, unsigned int x, unsigned int y, unsigned int z);
  void pointRemoved(unsigned int var, unsigned int time, int classIndex, unsigned int x, unsigned int y, unsigned int z);
  void depthChanged(int);

 protected:
  struct SliceTile
  {
    /* texture coords */
    float tex_coords[8]; /* 2 components per coordinate */
    GLuint texture;

    /* vertex coords (always within [-1.0,1.0]) */
    float vert_coords[8]; /* 2 components per coordinate */
  };
 
  unsigned char m_ByteMap[256*4]; /* The transfer function */
  unsigned char m_GreyMap[256*4]; /* a greyscale map */
  unsigned char *m_Palette; /* a pointer to one of the above maps */
  VolMagick::VolumeFileInfo m_VolumeFileInfo; // entire volume info
  VolMagick::Volume m_VolumeSlice; // contains current slice data
  boost::shared_array<unsigned char> m_Slice; // the slice as a unsigned char uploadable texture

  unsigned int m_Depth; /* current slice */

  QList<PointClass*> ***m_PointClassList; /* the list of point classes */
  PointClass *m_CurrentPointClass; /* the point class currently selected (for adding points) */
  int m_CurrentPointClassIndex; /* index of m_CurrentPointclass in m_PointClassList */

  SliceAxis m_SliceAxis; /* orientation of the slices */
  
  int m_PointSize; /* size of the markers drawn for each point */
  
  unsigned int m_Variable;
  unsigned int m_Timestep;
  
  bool m_Drawable; /* if this is false, the volume slices cannot be drawn */
  
  SliceTile *m_SliceTiles;
  unsigned int m_NumSliceTiles;
  
  /* texture and geometry dimensions */
  float m_TexCoordMinX, m_TexCoordMinY, m_TexCoordMaxX, m_TexCoordMaxY;
  float m_VertCoordMinX, m_VertCoordMinY, m_VertCoordMaxX, m_VertCoordMaxY;

  CellMarkingMode m_CellMarkingMode;

  /* the set of contours for the current loaded volume */
  SurfRecon::ContourPtrArray m_Contours;
  SurfRecon::ContourPtr m_CurrentContour;
  //int m_CurrentContourIndex; //index of m_CurrentContour in m_Contours;

  bool m_MouseIsDown;
  std::set<SurfRecon::WeakPointPtr> m_SelectedPoints;
  std::vector<SurfRecon::WeakPointPtr> m_AllPoints; //used for id <=> Point mapping... gets reset every time selection occurs

  QRect m_SelectionRectangle;

  // Different selection modes
  int m_SelectionMode;

  QPoint m_LastPoint;
  bool m_Drag;

  boost::scoped_ptr<SliceRenderer> m_SliceRenderer; //points to the calls to perform the actual rendering
  friend class ARBFragmentProgramSliceRenderer;
  friend class PalettedSliceRenderer;
  friend class SGIColorTableSliceRenderer;

  bool m_RenderSDF;
  SDF2D::ImagePtr m_SDF;

  bool m_RenderControlPoints;

  boost::scoped_ptr<qglviewer::WorldConstraint> m_Constraint;

  bool m_SliceDirty;
  bool m_MouseZoomStarted;
  bool m_UpdateSliceOnRelease;
  bool m_CameraInitialized;
};

class ARBFragmentProgramSliceRenderer : public SliceRenderer
{
 public:
  ARBFragmentProgramSliceRenderer(SliceCanvas *canvas);
  virtual ~ARBFragmentProgramSliceRenderer();

  virtual bool init();
  virtual void draw();
  virtual void uploadSlice(unsigned char *slice, unsigned int dimx, unsigned int dimy);
  virtual void uploadColorTable();
	
  GLuint m_FragmentProgram;
  GLuint m_PaletteTexture; /* texture id for the current palette */
};

class PalettedSliceRenderer : public SliceRenderer
{
 public:
  PalettedSliceRenderer(SliceCanvas *canvas);
  virtual ~PalettedSliceRenderer();

  virtual bool init();
  virtual void draw();
  virtual void uploadSlice(unsigned char *slice, unsigned int dimx, unsigned int dimy);
  virtual void uploadColorTable();
};

class SGIColorTableSliceRenderer : public SliceRenderer
{
 public:
  SGIColorTableSliceRenderer(SliceCanvas *canvas);
  virtual ~SGIColorTableSliceRenderer();

  virtual bool init();
  virtual void draw();
  virtual void uploadSlice(unsigned char *slice, unsigned int dimx, unsigned int dimy);
  virtual void uploadColorTable();
};

class VolumeGridRover : public QWidget 
{
  Q_OBJECT

    public:
  VolumeGridRover(QWidget* parent = nullptr, Qt::WindowFlags fl = Qt::WindowFlags());
  ~VolumeGridRover();

  SliceCanvas *getCurrentSliceCanvas();
  Ui::VolumeGridRoverBase *_ui;

  bool setVolume(const VolMagick::VolumeFileInfo& vfi);
  void unsetVolume();
  bool hasVolume(void);
 
  inline void setTransferFunction(unsigned char *func)
    {
      m_XYSliceCanvas->setTransferFunction(func);
      m_XZSliceCanvas->setTransferFunction(func);
      m_ZYSliceCanvas->setTransferFunction(func);
    }

   
 
  const QList<PointClass*>* getPointClassList() const;

  void addContour(const SurfRecon::Contour& c);
  void removeContour(const std::string& name);
		 
  unsigned int getXYDepth() const { return m_XYSliceCanvas->getDepth(); }
  unsigned int getXZDepth() const { return m_XZSliceCanvas->getDepth(); }
  unsigned int getZYDepth() const { return m_ZYSliceCanvas->getDepth(); }

  const SurfRecon::ContourPtrArray* getContours() const 
    { 
      return &m_Contours;
    }

  QString cacheDir() const;

  struct IsocontourValue
  {
    float value;
    float red;
    float green;
    float blue;
  };
  
  typedef std::vector<IsocontourValue> IsocontourValues;

  bool doIsocontouring() const { return m_DoIsocontouring; }
  const IsocontourValues& isocontourValues() const { return m_IsocontourValues; }
  void isocontourValues(const IsocontourValues&);

  void updateGL()
  {
    m_XYSliceCanvas->update();
    m_XZSliceCanvas->update();
    m_ZYSliceCanvas->update();
  }

  public slots:

  void getLocalOutputFileSlot();
  void getRemoteFileSlot();
  void localSegmentationRunSlot();
  void remoteSegmentationRunSlot();
  void sliceAxisChangedSlot();
  void EMClusteringRunSlot();
  void savePointClassesSlot();
  void loadPointClassesSlot();
  void saveContoursSlot();
  void loadContoursSlot();
  void sdfOptionsSlot();
  void medialAxisSlot();
  void curateContoursSlot();

  void colorSlot();

  void doIsocontouring(bool);
  void setCurrentVariable(int variable);
  void setCurrentTimestep(int timestep);
  void setCurrentData(int variable, int timestep);
  void setX(int x);
  void setY(int y);
  void setZ(int z);
  void setXYZ(int x, int y, int z);
  void setObjX(float objx);
  void setObjY(float objy);
  void setObjZ(float objz);
  void setObjXYZ(float objx, float objy, float objz);
  void setR(int r);
  void setG(int g);
  void setB(int b);
  void setA(int a);
  void setRGBA(int r, int g, int b, int a);
  void setValue(double value);
  void setMinValue(double value);
  void setMaxValue(double value);
  void setGridCellInfo(int x, int y, int z, 
		       float objx, float objy, float objz,
		       int r, int g, int b, int a, double value);
  void setPointSize(int r);
  void showColor(int i);
  void showContourColor(const QString& name);
  void showContourInterpolationType(const QString& name);
  void showContourInterpolationSampling(const QString& name);

  void currentObjectSelectionChanged();

  void setCellMarkingMode(int m);
  
  void ZDepthChangedSlot(int d);
  void YDepthChangedSlot(int d);
  void XDepthChangedSlot(int d);

  void setSelectedContours();

  void addPointClassSlot();
  void deletePointClassSlot();
  void xChangedSlot();
  void yChangedSlot();
  void zChangedSlot();
  void backgroundColorSlot();
  void addContourSlot();
  void deleteContourSlot();
  void contourColorSlot();
  void setInterpolationTypeSlot(int);
  void setInterpolationSamplingSlot(int);
  void tilingRunSlot();
  void getTilingOutputDirectorySlot();
  void handleTilingOutputDestinationSelectionSlot(int);
  void sdfCurationSlot();

 protected:
  void hideEvent(QHideEvent *e);
  void showEvent(QShowEvent *e);
  void customEvent(QEvent *e);
  void setColorName();
    	  
  //just emits the current page index of the current page in the cell
  //marking mode tab widget
  void cellMarkingModeTabChangedSlot(QWidget *w);

  void clearIsocontours();
  void generateIsocontours();

  void showCurrentObject();
  void updateSliceContours();

 signals:
  void showToggle(bool show);
  void pointAdded(unsigned int var, unsigned int time, int classIndex, unsigned int x, unsigned int y, unsigned int z);
  void pointRemoved(unsigned int var, unsigned int time, int classIndex, unsigned int x, unsigned int y, unsigned int z);
  void depthChanged(SliceCanvas *, int);
  void tilingComplete(const boost::shared_ptr<Geometry>&); //used to send tiled meshes to whoever cares
  void cellMarkingModeChanged(int);
  void showImage(const QImage& img);
  void volumeGenerated(const QString& vol_filename); //emitted if a temporary file was written

 private:
  /* The remote segmentation thread */
  class RemoteGenSegThread : public QThread
  {
  public:
    RemoteGenSegThread(VolumeGridRover *vgr, unsigned int stackSize = 0);
    virtual void run();

  private:
    VolumeGridRover *m_VolumeGridRover;
  };
  
  /* The local segmentation thread */
  class LocalGenSegThread : public QThread
  {
  public:
    LocalGenSegThread(VolumeGridRover *vgr, unsigned int stackSize = 0);
    virtual void run();

  private:
    VolumeGridRover *m_VolumeGridRover;
  };
	
  class TilingThread : public QThread
  {
  public:
    TilingThread(VolumeGridRover *vgr, unsigned int stackSize = 0);
    virtual void run();
    void setSelectedNames(const std::list<std::string>& names) { m_SelectedNames = names; }
  private:
    VolumeGridRover *m_VolumeGridRover;
    std::list<std::string> m_SelectedNames;
  };
	
  friend class RemoteGenSegThread;
  friend class LocalGenSegThread;
  friend class TilingThread;
  friend class PointClassFileContentHandler;
  friend class ContourFileContentHandler;
 
  SliceCanvas *m_XYSliceCanvas;
  SliceCanvas *m_XZSliceCanvas;
  SliceCanvas *m_ZYSliceCanvas;

  //eventually get rid of this BS and use the same idiom as what we did for m_Contours below
  QList<PointClass*> ***m_PointClassList;

  bool m_hasVolume;
  VolMagick::VolumeFileInfo m_VolumeFileInfo;

#ifdef USING_EM_CLUSTERING
  EM::cEMClustering<unsigned char> m_Clusters;
  int m_Histogram[256];
  bool m_HistogramCalculated;
#endif
  
  RemoteGenSegThread m_RemoteGenSegThread; /* Thread that handles the XmlRPC call to the remote general segmentation server */
  LocalGenSegThread m_LocalGenSegThread; /* Thread that handles local general segmentation */
  TilingThread m_TilingThread; /* Thread that calls tiling code */

  /* the set of contours for the current loaded volume */
  //std::vector<std::vector<std::vector<SurfRecon::ContourPtr> > > m_Contours;
  SurfRecon::ContourPtrArray m_Contours;

  CellMarkingMode m_CellMarkingMode;

  bool m_DoIsocontouring;
  // the set of values used for isocontour generation
  IsocontourValues m_IsocontourValues;
};

#endif
