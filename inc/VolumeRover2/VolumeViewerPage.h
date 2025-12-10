#ifndef __VOLUMEVIEWERPAGE_H__
#define __VOLUMEVIEWERPAGE_H__

//choose between the two how you want to extract isocontours
//#define ISOCONTOURING_WITH_LBIE
//#define ISOCONTOURING_WITH_FASTCONTOURING

//define the following if you want threaded isocontouring
//#define THREADED_ISOCONTOURING

#include <VolumeRover2/VolumeViewer.h>
#include <VolMagick/VolMagick.h>
#include <VolMagick/VolumeCache.h>

#ifdef ISOCONTOURING_WITH_LBIE
#include <LBIE/LBIE_Mesher.h>
#endif

#ifdef ISOCONTOURING_WITH_FASTCONTOURING
#include <FastContouring/FastContouring.h>
#endif

#include <ColorTable2/ColorTable.h>

#include <vector>
#include <map>

#include <qglobal.h>

#if QT_VERSION < 0x040000
# include <qwidget.h>
#else
# include <QWidget>
#endif

class QSlider;
class IsocontouringManager;
#if QT_VERSION < 0x040000
class QCustomEvent;
#else
class QEvent;
#endif

#if QT_VERSION > 0x040000
namespace Ui
{
  class VolumeViewerPage;
}
#endif

class VolumeViewerPage : public QWidget
{
  Q_OBJECT

 public:
  VolumeViewerPage(QWidget * parent = 0,
#if QT_VERSION < 0x040000
                   const char * name = 0
#else
                   Qt::WindowFlags flags = {}
#endif
                   );

  virtual ~VolumeViewerPage();

  CVC::VolumeViewer* thumbnailViewer() { return _thumbnailViewer; }
  CVC::VolumeViewer* subvolumeViewer() { return _subvolumeViewer; }

  CVCColorTable::ColorTable* colorTable() { return _colorTable; }

 public slots:
  virtual void openVolume(const VolMagick::VolumeFileInfo& vfi);
  virtual void close();

  //[min-max] of dataset
  void doGeometryExtraction(const CVCColorTable::isocontour_nodes& nodes,
                            const CVCColorTable::color_nodes& c_nodes);

 protected:
  void customEvent(
#if QT_VERSION < 0x040000
                   QCustomEvent *e
#else
                   QEvent *e
#endif
                   );

 protected slots:
  virtual void setDefaultSubVolumeViewerState();
  virtual void setDefaultThumbnailViewerState();
  virtual void setThumbnailQuality(int);
  virtual void setSubVolumeQuality(int);
  virtual void setThumbnailNearPlane(int);
  virtual void setSubVolumeNearPlane(int);

  virtual void extractNewSubVolume(const VolMagick::BoundingBox& subvolbox);
  virtual void updateColorTable();

 private:
  CVC::VolumeViewer *_thumbnailViewer;
  CVC::VolumeViewer *_subvolumeViewer;

  VolMagick::VolumeCache _volumeCache;

  QSlider *_subvolumeRenderQualitySlider;
  QSlider *_subvolumeNearClipPlaneSlider;
  QSlider *_thumbnailRenderQualitySlider;
  QSlider *_thumbnailNearClipPlaneSlider;

  CVCColorTable::ColorTable *_colorTable;

  //Object that handles the VolumeViewerPage's isocontouring support
  IsocontouringManager *_isocontouring;

  typedef std::map<
    VolMagick::BoundingBox,
    cvcraw_geometry::geometry_t
    > isogeom_t;

  typedef std::map<
    double, //isovalue
    isogeom_t
    > isocache_t;

  isocache_t _isocache;

#if QT_VERSION > 0x040000
  Ui::VolumeViewerPage *_ui;
#endif
};

#endif
