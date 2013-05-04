#include <VolumeRover2/VolumeViewerPage.h>

#include <qglobal.h>

#if QT_VERSION < 0x040000
# include <qprogressdialog.h>
# include <qstring.h>
# include <qcolor.h>
# include <qlayout.h>
# include <qhbox.h>
# include <qvbox.h>
# include <qpopupmenu.h>
# include <qmenubar.h>
# include <qaction.h>
# include <qfiledialog.h>
# include <qmessagebox.h>
# include <qlabel.h>
# include <qslider.h>
# include <qapplication.h>
# ifdef THREADED_ISOCONTOURING
#  include <qthread.h>
# endif
#else
# include <QApplication>
# include <QWidget>
# include <QMainWindow>
# include <QProgressDialog>
# include <QString>
# include <QColor>
# include <QHBoxLayout>
# include <QVBoxLayout>
# include <QEvent>
# include <QSlider>
# include <QMessageBox>
# ifdef THREADED_ISOCONTOURING
#  include <QThread>
# endif
#endif

#include <cstring>
#include <list>
#include <boost/utility.hpp>
#include <boost/shared_array.hpp>
#include <boost/tuple/tuple.hpp>

#if QT_VERSION > 0x040000
#include "ui_VolumeViewerPage.h"
#endif

namespace
{
#ifdef ISOCONTOURING_WITH_LBIE
  //utility function to convert from LBIE geoframe to cvcraw geometry
  cvcraw_geometry::geometry_t convert(const LBIE::geoframe& geo)
  {
    using namespace std;
    cvcraw_geometry::geometry_t ret_geom;
    ret_geom.points.resize(geo.verts.size());
    copy(geo.verts.begin(),
	 geo.verts.end(),
	 ret_geom.points.begin());
    ret_geom.normals.resize(geo.normals.size());
    copy(geo.normals.begin(),
	 geo.normals.end(),
	 ret_geom.normals.begin());
    ret_geom.colors.resize(geo.color.size());
    copy(geo.color.begin(),
	 geo.color.end(),
	 ret_geom.colors.begin());
    ret_geom.boundary.resize(geo.bound_sign.size());
    for(vector<unsigned int>::const_iterator j = geo.bound_sign.begin();
	j != geo.bound_sign.end();
	j++)
      ret_geom.boundary[distance(j,geo.bound_sign.begin())] = *j;
    ret_geom.tris.resize(geo.triangles.size());
    copy(geo.triangles.begin(),
	 geo.triangles.end(),
	 ret_geom.tris.begin());
    ret_geom.quads.resize(geo.quads.size());
    copy(geo.quads.begin(),
	 geo.quads.end(),
	 ret_geom.quads.begin());
    return ret_geom;
  }
#endif

#ifdef ISOCONTOURING_WITH_FASTCONTOURING
  cvcraw_geometry::geometry_t convert(const FastContouring::TriSurf& geo)
  {
    using namespace std;
    cvcraw_geometry::geometry_t ret_geom;
    ret_geom.points.resize(geo.verts.size()/3);
    memcpy(&(ret_geom.points[0]),
	   &(geo.verts[0]),
	   geo.verts.size()*sizeof(double));
    ret_geom.normals.resize(geo.normals.size()/3);
    memcpy(&(ret_geom.normals[0]),
	   &(geo.normals[0]),
	   geo.normals.size()*sizeof(double));
    ret_geom.colors.resize(geo.colors.size()/3);
    memcpy(&(ret_geom.colors[0]),
	   &(geo.colors[0]),
	   geo.colors.size()*sizeof(double));
    ret_geom.tris.resize(geo.tris.size()/3);
    memcpy(&(ret_geom.tris[0]),
	   &(geo.tris[0]),
	   geo.tris.size()*sizeof(unsigned int));
    return ret_geom;
  }
#endif

  const double MIN_RANGE = 0.0;
  const double MAX_RANGE = 1.0;
  template <typename T> static inline T clamp(T val)
  {
    return std::max(MIN_RANGE,std::min(MAX_RANGE,val));
  }

#ifdef THREADED_ISOCONTOURING
  class IsocontourThreadFinishedEvent : 
#if QT_VERSION < 0x040000
    public QCustomEvent
#else
    public QEvent
#endif
  {
  public:
    typedef boost::tuple<cvcraw_geometry::geometry_t,
			 cvcraw_geometry::geometry_t> isocontour_geometry;
    IsocontourThreadFinishedEvent
    (const isocontour_geometry& geoms)
      :
#if QT_VERSION < 0x040000 
      QCustomEvent(31337),
#else
      QEvent(QEvent::Type(31337)),
#endif 
      _output(geoms) {}
    
    isocontour_geometry& output() { return _output; }
    const isocontour_geometry& output() const { return _output; }

  private:
    isocontour_geometry _output;
  };

  class IsocontourThreadFailedEvent : 
#if QT_VERSION < 0x040000
    public QCustomEvent
#else
    public QEvent
#endif
  {
  public:
    IsocontourThreadFailedEvent(const QString& m)
      :
#if QT_VERSION < 0x040000 
      QCustomEvent(31338),
#else
      QEvent(QEvent::Type(31338)),
#endif 
      _msg(m) {}
    QString message() const { return _msg; }
  private:
    QString _msg;
  };
#endif
}

class IsocontouringManager
{
public:
  IsocontouringManager(VolumeViewerPage *vvp)
#ifdef THREADED_ISOCONTOURING
    : _isocontourExtractionThread(vvp)
#endif
  {
  }

  ~IsocontouringManager()
  {
#ifdef THREADED_ISOCONTOURING
    _isocontourExtractionThread.wait();
#endif
  }

#ifdef THREADED_ISOCONTOURING
  class IsocontourExtractionThread : public QThread
  {
  public:
    IsocontourExtractionThread(VolumeViewerPage *vvp, unsigned int stackSize = 0);
    virtual void run();
    void setNodes(const CVCColorTable::isocontour_nodes& n,
		  const CVCColorTable::color_nodes& c);
    const CVCColorTable::isocontour_nodes& nodes() const { return _nodes; }
    const CVCColorTable::color_nodes& c_nodes() const { return _c_nodes; }
    void clearNodes();
  private:
    CVCColorTable::isocontour_nodes _nodes;
    CVCColorTable::color_nodes _c_nodes;
    VolumeViewerPage *_volumeViewerPage;
  };
#endif

#ifdef THREADED_ISOCONTOURING
  IsocontourExtractionThread& isocontourExtractionThread()
  { return _isocontourExtractionThread; }
#endif
  
#ifdef ISOCONTOURING_WITH_LBIE
  std::vector<LBIE::Mesher>& thumbnailMeshers() 
  { return _thumbnailMeshers; }
  std::vector<LBIE::Mesher>& subvolumeMeshers()
  { return _subvolumeMeshers; }
#endif

#ifdef ISOCONTOURING_WITH_FASTCONTOURING
  FastContouring::ContourExtractor& thumbnailContourExtractor()
  { return _thumbnailContourExtractor; }
  FastContouring::ContourExtractor& subvolumeContourExtractor()
  { return _subvolumeContourExtractor; }
#endif

private:
#ifdef THREADED_ISOCONTOURING
  IsocontourExtractionThread _isocontourExtractionThread;
#endif

#ifdef ISOCONTOURING_WITH_LBIE
  //for mesh extraction, 1 mesher per isocontour bar
  std::vector<LBIE::Mesher> _thumbnailMeshers;
  std::vector<LBIE::Mesher> _subvolumeMeshers;
#endif

#ifdef ISOCONTOURING_WITH_FASTCONTOURING
  FastContouring::ContourExtractor _thumbnailContourExtractor;
  FastContouring::ContourExtractor _subvolumeContourExtractor;
#endif
};

VolumeViewerPage::VolumeViewerPage(QWidget * parent,
#if QT_VERSION < 0x040000 
				   const char * name
#else
                                   Qt::WFlags flags
#endif
)
  : QWidget(parent,
#if QT_VERSION < 0x040000 
            name
#else
            flags
#endif
            ),
    _thumbnailViewer(NULL),
    _subvolumeViewer(NULL),
    _subvolumeRenderQualitySlider(NULL),
    _subvolumeNearClipPlaneSlider(NULL),
    _thumbnailRenderQualitySlider(NULL),
    _thumbnailNearClipPlaneSlider(NULL),
    _colorTable(NULL),
    _isocontouring(NULL)
{
  _isocontouring = new IsocontouringManager(this);

#if QT_VERSION < 0x040000
  if(!name)
    setName("VolumeViewerPage");
#endif
    
  //set up the main central widget
#if QT_VERSION < 0x040000
  QGridLayout *mainLayout = new QGridLayout(this,2,1);
  QHBox *hbox_left_and_right = new QHBox(this);
  hbox_left_and_right->setMargin(3);
  hbox_left_and_right->setSpacing(3);
  //setCentralWidget(hbox_left_and_right);
  mainLayout->addWidget(hbox_left_and_right,0,0);

  QVBox *vbox_left = new QVBox(/*centralWidget()*/ hbox_left_and_right);
  {
    _subvolumeViewer = new VolumeViewer::Viewer(vbox_left);
    QHBox *hbox_subvolume_render_quality_group = new QHBox(vbox_left);
    hbox_subvolume_render_quality_group->setSizePolicy(QSizePolicy::Preferred,
						       QSizePolicy::Fixed);
    {
      new QLabel("Sub-volume render quality", hbox_subvolume_render_quality_group);
      _subvolumeRenderQualitySlider = new QSlider(0,255,5,255,Qt::Horizontal,
						  hbox_subvolume_render_quality_group);
      _subvolumeRenderQualitySlider->setSizePolicy(QSizePolicy::MinimumExpanding,
						   QSizePolicy::Fixed);
      connect(_subvolumeRenderQualitySlider,
	      SIGNAL(valueChanged(int)),
	      SLOT(setSubVolumeQuality(int)));
      connect(_subvolumeRenderQualitySlider,
	      SIGNAL(sliderMoved(int)),
	      SLOT(setSubVolumeQuality(int)));
    }
    QHBox *hbox_subvolume_near_clip_plane_group = new QHBox(vbox_left);
    hbox_subvolume_near_clip_plane_group->setSizePolicy(QSizePolicy::Preferred,
							QSizePolicy::Fixed);
    {
      new QLabel("Sub-volume near clip plane", hbox_subvolume_near_clip_plane_group);
      _subvolumeNearClipPlaneSlider = new QSlider(0,255,5,0,Qt::Horizontal,
						  hbox_subvolume_near_clip_plane_group);
      _subvolumeNearClipPlaneSlider->setSizePolicy(QSizePolicy::MinimumExpanding,
						   QSizePolicy::Fixed);
      connect(_subvolumeNearClipPlaneSlider,
	      SIGNAL(valueChanged(int)),
	      SLOT(setSubVolumeNearPlane(int)));
      connect(_subvolumeNearClipPlaneSlider,
	      SIGNAL(sliderMoved(int)),
	      SLOT(setSubVolumeNearPlane(int)));
    }
  }

  QVBox *vbox_right = new QVBox(/*centralWidget()*/ hbox_left_and_right);
  {
    _thumbnailViewer = new VolumeViewer::Viewer(vbox_right);
    QHBox *hbox_thumbnail_render_quality_group = new QHBox(vbox_right);
    hbox_thumbnail_render_quality_group->setSizePolicy(QSizePolicy::Preferred,
						       QSizePolicy::Fixed);
    {
      new QLabel("Thumbnail render quality", hbox_thumbnail_render_quality_group);
      _thumbnailRenderQualitySlider = new QSlider(0,255,5,255,Qt::Horizontal,
						  hbox_thumbnail_render_quality_group);
      _thumbnailRenderQualitySlider->setSizePolicy(QSizePolicy::MinimumExpanding,
						   QSizePolicy::Fixed);
      connect(_thumbnailRenderQualitySlider,
	      SIGNAL(valueChanged(int)),
	      SLOT(setThumbnailQuality(int)));
      connect(_thumbnailRenderQualitySlider,
	      SIGNAL(sliderMoved(int)),
	      SLOT(setThumbnailQuality(int)));
    }
    QHBox *hbox_thumbnail_near_clip_plane_group = new QHBox(vbox_right);
    hbox_thumbnail_near_clip_plane_group->setSizePolicy(QSizePolicy::Preferred,
							QSizePolicy::Fixed);
    {
      new QLabel("Thumbnail near clip plane", hbox_thumbnail_near_clip_plane_group);
      _thumbnailNearClipPlaneSlider = new QSlider(0,255,5,0,Qt::Horizontal,
						  hbox_thumbnail_near_clip_plane_group);
      _thumbnailNearClipPlaneSlider->setSizePolicy(QSizePolicy::MinimumExpanding,
						   QSizePolicy::Fixed);
      connect(_thumbnailNearClipPlaneSlider,
	      SIGNAL(valueChanged(int)),
	      SLOT(setThumbnailNearPlane(int)));
      connect(_thumbnailNearClipPlaneSlider,
	      SIGNAL(sliderMoved(int)),
	      SLOT(setThumbnailNearPlane(int)));
    }
  }

#else
  _ui = new Ui::VolumeViewerPage;
  _ui->setupUi(this);
  _subvolumeViewer = new VolumeViewer::Viewer(_ui->_subvolumeViewerFrame, NULL, flags);
  _thumbnailViewer = new VolumeViewer::Viewer(_ui->_thumbnailViewerFrame, NULL, flags);
  QGridLayout *subvolumeViewerFrameLayout = new QGridLayout(_ui->_subvolumeViewerFrame);
  subvolumeViewerFrameLayout->addWidget(_subvolumeViewer,0,0);
  QGridLayout *thumbnailViewerFrameLayout = new QGridLayout(_ui->_thumbnailViewerFrame);
  thumbnailViewerFrameLayout->addWidget(_thumbnailViewer,0,0);
  _subvolumeRenderQualitySlider = _ui->_subvolumeRenderQualitySlider;
  _subvolumeNearClipPlaneSlider = _ui->_subvolumeNearClipPlaneSlider;
  _thumbnailRenderQualitySlider = _ui->_thumbnailRenderQualitySlider;
  _thumbnailNearClipPlaneSlider = _ui->_thumbnailNearClipPlaneSlider;
  connect(_subvolumeRenderQualitySlider,
          SIGNAL(valueChanged(int)),
          SLOT(setSubVolumeQuality(int)));
  connect(_subvolumeRenderQualitySlider,
          SIGNAL(sliderMoved(int)),
          SLOT(setSubVolumeQuality(int)));
  connect(_subvolumeNearClipPlaneSlider,
          SIGNAL(valueChanged(int)),
          SLOT(setSubVolumeNearPlane(int)));
  connect(_subvolumeNearClipPlaneSlider,
          SIGNAL(sliderMoved(int)),
          SLOT(setSubVolumeNearPlane(int)));
  connect(_thumbnailRenderQualitySlider,
          SIGNAL(valueChanged(int)),
          SLOT(setThumbnailQuality(int)));
  connect(_thumbnailRenderQualitySlider,
          SIGNAL(sliderMoved(int)),
          SLOT(setThumbnailQuality(int)));
  connect(_thumbnailNearClipPlaneSlider,
          SIGNAL(valueChanged(int)),
          SLOT(setThumbnailNearPlane(int)));
  connect(_thumbnailNearClipPlaneSlider,
          SIGNAL(sliderMoved(int)),
          SLOT(setThumbnailNearPlane(int)));
#endif

  //postInit() is useful for resetting the viewer state to something known
  //after a volume has been opened or closed...
  connect(_subvolumeViewer,SIGNAL(postInit()),SLOT(setDefaultSubVolumeViewerState()));
  connect(_thumbnailViewer,SIGNAL(postInit()),SLOT(setDefaultThumbnailViewerState()));
  connect(_thumbnailViewer,
	  SIGNAL(subVolumeSelectorChanged(const VolMagick::BoundingBox&)),
	  SLOT(extractNewSubVolume(const VolMagick::BoundingBox&)));

  //add the color table
#if QT_VERSION < 0x040000
  _colorTable = new CVCColorTable::ColorTable(this,"ColorTable");
  mainLayout->addWidget(_colorTable,1,0);
#else
  _colorTable = new CVCColorTable::ColorTable(_ui->_colortableFrame);
  QGridLayout *colorTableFrameLayout = new QGridLayout(_ui->_colortableFrame);
  colorTableFrameLayout->addWidget(_colorTable,0,0);
#endif
  connect(_colorTable,SIGNAL(changed()),SLOT(updateColorTable()));
#ifdef THREADED_ISOCONTOURING
  _colorTable->interactiveUpdates(true);
#else
  _colorTable->interactiveUpdates(false);
#endif

  //initialize the color tables in the viewer so 
  //when a volume loads it doesn't show up white at first!
  updateColorTable();
}

VolumeViewerPage::~VolumeViewerPage()
{
  delete _isocontouring;
#if QT_VERSION > 0x040000
  delete _ui;
#endif
}

void VolumeViewerPage::openVolume(const VolMagick::VolumeFileInfo& vfi)
{
  qDebug("VolumeViewerPage::openVolume()");
#ifdef THREADED_ISOCONTOURING
  _isocontouring->isocontourExtractionThread().wait();
#endif

  try
    {
      //_volumeCache = buildCache(vfi,"/tmp/transfix/testcache");

      _volumeCache.clear();
      _volumeCache.add(vfi);

      VolMagick::Volume thumb_vol(_volumeCache.get(vfi.boundingBox()));
      _thumbnailViewer->colorMappedVolume(thumb_vol);
      _thumbnailViewer->volumeRenderingType(VolumeViewer::Viewer::ColorMapped);
      VolMagick::Volume sub_vol(_volumeCache.get(_thumbnailViewer->subVolumeSelector()));
      _subvolumeViewer->colorMappedVolume(sub_vol);
      _subvolumeViewer->volumeRenderingType(VolumeViewer::Viewer::ColorMapped);

#ifdef ISOCONTOURING_WITH_FASTCONTOURING
      _isocontouring->thumbnailContourExtractor().setVolume(thumb_vol);
      _isocontouring->subvolumeContourExtractor().setVolume(sub_vol);
#endif

#ifdef ISOCONTOURING_WITH_LBIE
      for(std::vector<LBIE::Mesher>::iterator i = _isocontouring->thumbnailMeshers().begin();
	  i != _isocontouring->thumbnailMeshers().end();
	  i++)
	i->setVolume(thumb_vol);
      for(std::vector<LBIE::Mesher>::iterator i = _isocontouring->subvolumeMeshers().begin();
	  i != _isocontouring->subvolumeMeshers().end();
	  i++)
	i->setVolume(sub_vol);
#endif

      //extract any contours if necessary and update the scene.
#ifdef THREADED_ISOCONTOURING
      _isocontouring->isocontourExtractionThread().clearNodes();
#endif

      //_colorTable->setContourVolume(sub_vol);
      _colorTable->setContourVolume(thumb_vol);

      updateColorTable();
    }
  catch(VolMagick::Exception &e)
    {
      QMessageBox::critical(this,
			    "Error Loading File",
			    QString("%1").arg(e.what()),
			    QMessageBox::Ok,
			    QMessageBox::NoButton);
    }
}

void VolumeViewerPage::close()
{
#ifdef THREADED_ISOCONTOURING
  _isocontouring->isocontourExtractionThread().wait();
#endif

  _subvolumeViewer->reset();
  _thumbnailViewer->reset();
}

void VolumeViewerPage::doGeometryExtraction(const CVCColorTable::isocontour_nodes& nodes,
                                            const CVCColorTable::color_nodes& c_nodes)
{
  //using namespace std;
  using namespace boost;
  using namespace CVCColorTable;

  /*
    This function relies on:
    _volumeCache, _thumbnailViewer, 
    _isocontouring->thumbnailMeshers(), _isocontouring->subvolumeMeshers(),
    _isocontouring->thumbnailContourExtractor(), _isocontouring->subvolumeContourExtractor()
    _isocache

    If the main thread wants to access any of those variables and this function
    is being executed in a separate thread from the main thread, it should first
    call wait() on this thread...
  */

  try
    {
      if(!nodes.empty())
	{
	  cvcraw_geometry::geometry_t thumbnail_iso_geom;
	  cvcraw_geometry::geometry_t subvolume_iso_geom;

          std::list<double> needed_isovals;
          std::list<double> erase_isovals;
	  for(isocache_t::iterator i = _isocache.begin();
	      i != _isocache.end();
	      i++)
	    {
	      if(nodes.find(CVCColorTable::isocontour_node(i->first)) ==
		 nodes.end())
		erase_isovals.push_back(i->first);
	      
	    }

#ifdef ISOCONTOURING_WITH_LBIE
	  VolMagick::Volume vol(_volumeCache.get(_volumeCache.boundingBox()));
	  LBIE::Mesher def_thumb_mesher;
	  def_thumb_mesher.extractionMethod(LBIE::Mesher::FASTCONTOURING);
	  def_thumb_mesher.setVolume(vol);
	  _isocontouring->thumbnailMeshers().resize(nodes.size(),def_thumb_mesher);
	  LBIE::Mesher def_subvol_mesher;
	  def_subvol_mesher.extractionMethod(LBIE::Mesher::FASTCONTOURING);
	  def_subvol_mesher.setVolume(_volumeCache.get(_thumbnailViewer->subVolumeSelector()));
	  _isocontouring->subvolumeMeshers().resize(nodes.size(),def_subvol_mesher);
          int node_idx = 0;
	  for(isocontour_nodes::const_iterator i = nodes.begin();
	      i != nodes.end();
	      i++)
	    {
	      double isoval = vol.min()+(vol.max()-vol.min())*i->position;
	      cvcraw_geometry::geometry_t cur_thumb_geom = 
		_isocache[isoval][_volumeCache.boundingBox()];
	      cvcraw_geometry::geometry_t cur_sub_geom = 
		_isocache[isoval][_thumbnailViewer->subVolumeSelector()];

	      if(cur_thumb_geom.empty())
		{
		  //put thumbnail isocontour calculation here
		  _isocontouring->thumbnailMeshers()[node_idx].isovalue(isoval);
		  LBIE::geoframe &extracted_thumb =
		    _isocontouring->thumbnailMeshers()[node_idx].extractMesh();
		  cur_thumb_geom = convert(extracted_thumb);
		}
	      if(cur_sub_geom.empty())
		{
		  //put subvolume isocontour calculation here
		  _isocontouring->subvolumeMeshers()[node_idx].isovalue(isoval);
		  LBIE::geoframe &extracted_sub =
		    _isocontouring->subvolumeMeshers()[node_idx].extractMesh();
		  cur_sub_geom = convert(extracted_sub);
		}
              node_idx++;
	      
	      //calculate a color for the isocontour via the color nodes
	      double r = 1.0, g = 1.0, b = 1.0;
	      if(c_nodes.size() >= 2)
		{
		  color_nodes::const_iterator low_itr;
		  color_nodes::const_iterator high_itr = 
		    c_nodes.lower_bound(color_node(i->position));
		  if(high_itr != c_nodes.end())
		    {
		      low_itr =
			high_itr == c_nodes.begin() ? high_itr : prior(high_itr);
		    }
		  color_node high = *high_itr;
		  color_node low = *low_itr;
		  double interval_pos = 
		    (i->position - low.position)/
		    (high.position - low.position);
		  r = clamp(low.r + (high.r - low.r)*interval_pos);
		  g = clamp(low.g + (high.g - low.g)*interval_pos);
		  b = clamp(low.b + (high.b - low.b)*interval_pos);
		}
	      
	      cvcraw_geometry::geometry_t::color_t color = {{ r,g,b }};
	      cur_thumb_geom.colors.resize(cur_thumb_geom.points.size());
	      fill(cur_thumb_geom.colors.begin(),
		   cur_thumb_geom.colors.end(),
		   color);
	      cur_sub_geom.colors.resize(cur_sub_geom.points.size());
	      fill(cur_sub_geom.colors.begin(),
		   cur_sub_geom.colors.end(),
		   color);
	      thumbnail_iso_geom.merge(cur_thumb_geom);
	      subvolume_iso_geom.merge(cur_sub_geom);
	    }
#endif

#ifdef ISOCONTOURING_WITH_FASTCONTOURING
	  VolMagick::Volume vol(_volumeCache.get(_volumeCache.boundingBox()));
	  for(isocontour_nodes::const_iterator i = nodes.begin();
	      i != nodes.end();
	      i++)
	    {
	      double isoval = vol.min()+(vol.max()-vol.min())*i->position;
	      cvcraw_geometry::geometry_t cur_thumb_geom = 
		_isocache[isoval][_volumeCache.boundingBox()];
	      cvcraw_geometry::geometry_t cur_sub_geom = 
		_isocache[isoval][_thumbnailViewer->subVolumeSelector()];

	      //calculate a color for the isocontour via the color nodes
	      double r = 1.0, g = 1.0, b = 1.0;
	      if(c_nodes.size() >= 2)
		{
		  color_nodes::const_iterator low_itr;
		  color_nodes::const_iterator high_itr = 
		    c_nodes.lower_bound(color_node(i->position));
		  if(high_itr != c_nodes.end())
		    {
		      low_itr =
			high_itr == c_nodes.begin() ? high_itr : prior(high_itr);
		    }
		  color_node high = *high_itr;
		  color_node low = *low_itr;
		  double interval_pos = 
		    (i->position - low.position)/
		    (high.position - low.position);
		  r = clamp(low.r + (high.r - low.r)*interval_pos);
		  g = clamp(low.g + (high.g - low.g)*interval_pos);
		  b = clamp(low.b + (high.b - low.b)*interval_pos);
		}

	      if(cur_thumb_geom.empty())
		cur_thumb_geom = 
		  convert(
		    _isocontouring->thumbnailContourExtractor().extractContour(isoval,r,g,b)
		    );
	      else
		{
		  //just fill in the color
		  cvcraw_geometry::geometry_t::color_t color = {{ r,g,b }};
		  cur_thumb_geom.colors.resize(cur_thumb_geom.points.size());
		  fill(cur_thumb_geom.colors.begin(),
		       cur_thumb_geom.colors.end(),
		       color);
		}

	      if(cur_sub_geom.empty())
		cur_sub_geom = 
		  convert(
		    _isocontouring->subvolumeContourExtractor().extractContour(isoval,r,g,b)
		    );
	      else
		{
		  //just fill in the color
		  cvcraw_geometry::geometry_t::color_t color = {{ r,g,b }};
		  cur_sub_geom.colors.resize(cur_sub_geom.points.size());
		  fill(cur_sub_geom.colors.begin(),
		       cur_sub_geom.colors.end(),
		       color);
		}

	      thumbnail_iso_geom.merge(cur_thumb_geom);
	      subvolume_iso_geom.merge(cur_sub_geom);
	    }
#endif

#ifdef THREADED_ISOCONTOURING
	  QApplication::postEvent(this,
				  new IsocontourThreadFinishedEvent(make_tuple(thumbnail_iso_geom,
									       subvolume_iso_geom)));
#else
          //Add the geometry immediately if not using threading.
          _thumbnailViewer->addGeometry(thumbnail_iso_geom,"isocontour");
          _subvolumeViewer->addGeometry(subvolume_iso_geom,"isocontour");
#endif
	}
      else
	{
	  //just overwrite it with empty geometry for now
#ifdef THREADED_ISOCONTOURING
	  QApplication::postEvent(this,
				  new IsocontourThreadFinishedEvent(make_tuple(cvcraw_geometry::geometry_t(),
									       cvcraw_geometry::geometry_t())));
#else
          //Add the geometry immediately if not using threading.
          _thumbnailViewer->addGeometry(cvcraw_geometry::geometry_t(),"isocontour");
          _subvolumeViewer->addGeometry(cvcraw_geometry::geometry_t(),"isocontour");
#endif
	}
    }
  catch(std::exception& e)
    {
#ifdef THREADED_ISOCONTOURING
      QApplication::postEvent(this,
			      new IsocontourThreadFailedEvent(QString("Error: %1").arg(e.what())));
#else
      qDebug("VolumeViewerPage::doGeometryExtraction: Error: %s",e.what());
#endif
    }
}

void VolumeViewerPage::customEvent(
#if QT_VERSION < 0x040000
                                   QCustomEvent *e
#else
                                   QEvent *e
#endif
                                   )
{
  using namespace boost;
  using namespace CVCColorTable;

#ifdef THREADED_ISOCONTOURING
  if(e->type() == 31337) //if the event type is our IsocontourThreadFinishedEvent
    {
      typedef VolumeViewer::scene_geometry_ptr scene_geometry_ptr;
      typedef VolumeViewer::scene_geometry_t   scene_geometry_t;
      IsocontourThreadFinishedEvent *itfe = static_cast<IsocontourThreadFinishedEvent*>(e);
      cvcraw_geometry::geometry_t& thumbnail_iso_geom = itfe->output().get<0>();
      cvcraw_geometry::geometry_t& subvolume_iso_geom = itfe->output().get<1>();

      _thumbnailViewer->addGeometry(thumbnail_iso_geom,"isocontour");
      _subvolumeViewer->addGeometry(subvolume_iso_geom,"isocontour");

      //if the thread's set of nodes now differs with what is in the color table,
      //we must re-run the thread to calculate the new isocontours...
      const isocontour_nodes& nodes = _colorTable->info().isocontourNodes();
      const color_nodes& c_nodes = _colorTable->info().colorNodes();
      if(nodes != _isocontouring->isocontourExtractionThread().nodes() ||
	 c_nodes != _isocontouring->isocontourExtractionThread().c_nodes())
	{
	  _isocontouring->isocontourExtractionThread().setNodes(nodes,c_nodes);
	  _isocontouring->isocontourExtractionThread().start();
	}
    }
  else if(e->type() == 31338) //IsocontourThreadFailedEvent
    {
      IsocontourThreadFailedEvent *itfe = static_cast<IsocontourThreadFailedEvent*>(e);
#if QT_VERSION < 0x040000
      std::string message(itfe->message().ascii());
#else
      std::string message((const char *)itfe->message().toAscii());
#endif
      qDebug("VolumeViewerPage::customEvent(): Received IsocontourThreadFailedEvent: %s",
	     message.c_str());
    }
#endif
}

void VolumeViewerPage::setDefaultSubVolumeViewerState()
{
  _subvolumeViewer->drawBoundingBox(true);
  //_subvolumeViewer->drawSubVolumeSelector(true);
  _subvolumeViewer->setBackgroundColor(QColor(0,0,0));
  _subvolumeViewer->drawCornerAxis(true);
  _subvolumeViewer->showEntireScene();
}

void VolumeViewerPage::setDefaultThumbnailViewerState()
{
  _thumbnailViewer->drawBoundingBox(true);
  _thumbnailViewer->drawSubVolumeSelector(true);
  _thumbnailViewer->setBackgroundColor(QColor(0,0,0));
  _thumbnailViewer->drawCornerAxis(true);
  _thumbnailViewer->showEntireScene();
}

void VolumeViewerPage::setThumbnailQuality(int q)
{
#if QT_VERSION < 0x040000
  _thumbnailViewer->quality(float(q-_thumbnailRenderQualitySlider->minValue())/
			    float(_thumbnailRenderQualitySlider->maxValue()-
				  _thumbnailRenderQualitySlider->minValue()));
#else
  _thumbnailViewer->quality(float(q-_thumbnailRenderQualitySlider->minimum())/
			    float(_thumbnailRenderQualitySlider->maximum()-
				  _thumbnailRenderQualitySlider->minimum()));
#endif
}

void VolumeViewerPage::setSubVolumeQuality(int q)
{
#if QT_VERSION < 0x040000
  _subvolumeViewer->quality(float(q-_subvolumeRenderQualitySlider->minValue())/
			    float(_subvolumeRenderQualitySlider->maxValue()-
				  _subvolumeRenderQualitySlider->minValue()));
#else
  _subvolumeViewer->quality(float(q-_subvolumeRenderQualitySlider->minimum())/
			    float(_subvolumeRenderQualitySlider->maximum()-
				  _subvolumeRenderQualitySlider->minimum()));
#endif
}

void VolumeViewerPage::setThumbnailNearPlane(int q)
{
#if QT_VERSION < 0x040000
  _thumbnailViewer->nearPlane(float(q-_thumbnailNearClipPlaneSlider->minValue())/
			      float(_thumbnailNearClipPlaneSlider->maxValue()-
				    _thumbnailNearClipPlaneSlider->minValue()));
#else
  _thumbnailViewer->nearPlane(float(q-_thumbnailNearClipPlaneSlider->minimum())/
			      float(_thumbnailNearClipPlaneSlider->maximum()-
				    _thumbnailNearClipPlaneSlider->minimum()));
#endif
}

void VolumeViewerPage::setSubVolumeNearPlane(int q)
{
#if QT_VERSION < 0x040000
  _subvolumeViewer->nearPlane(float(q-_subvolumeNearClipPlaneSlider->minValue())/
			      float(_subvolumeNearClipPlaneSlider->maxValue()-
				    _subvolumeNearClipPlaneSlider->minValue()));
#else
  _subvolumeViewer->nearPlane(float(q-_subvolumeNearClipPlaneSlider->minimum())/
			      float(_subvolumeNearClipPlaneSlider->maximum()-
				    _subvolumeNearClipPlaneSlider->minimum()));
#endif
}

void VolumeViewerPage::extractNewSubVolume(const VolMagick::BoundingBox& subvolbox)
{
#ifdef THREADED_ISOCONTOURING
  _isocontouring->isocontourExtractionThread().wait();
#endif

  try
    {
      VolMagick::Volume sub_vol(_volumeCache.get(subvolbox));
      _subvolumeViewer->colorMappedVolume(sub_vol);
      //_colorTable->setContourVolume(sub_vol);

#ifdef ISOCONTOURING_WITH_LBIE
      for(std::vector<LBIE::Mesher>::iterator i = _isocontouring->subvolumeMeshers().begin();
	  i != _isocontouring->subvolumeMeshers().end();
	  i++)
	i->setVolume(_volumeCache.get(subvolbox));
#endif

#ifdef ISOCONTOURING_WITH_FASTCONTOURING
      _isocontouring->subvolumeContourExtractor()
	.setVolume(_volumeCache.get(subvolbox));
#endif

#ifdef __GNUC__
#warning extractNewSubVolume(): consider having a separate thread for subvolume isocontour generation
#endif
      //it would be nice to be able to avoid having to recalculate the thumbnail isocontour

      //re-extract any contours if necessary and update the scene.
#ifdef THREADED_ISOCONTOURING
      _isocontouring->isocontourExtractionThread().clearNodes();
#endif
      updateColorTable();
    }
  catch(VolMagick::Exception &e)
    {
      qDebug("Warning: %s",e.what());
    }
}

void VolumeViewerPage::updateColorTable()
{
  using namespace boost;
  using namespace CVCColorTable;

  typedef VolumeViewer::scene_geometry_ptr scene_geometry_ptr;
  typedef VolumeViewer::scene_geometry_t   scene_geometry_t;

  const isocontour_nodes& nodes = _colorTable->info().isocontourNodes();
  const color_nodes& c_nodes = _colorTable->info().colorNodes();

#ifndef THREADED_ISOCONTOURING
  doGeometryExtraction(nodes,c_nodes);
#else
  //start up a new isocontouring thread if we dont have one running,
  //and if the set of nodes we care about differs from what's in the
  //thread storage
#if QT_VERSION < 0x040000
  bool running = _isocontouring->isocontourExtractionThread().running();
#else
  bool running = _isocontouring->isocontourExtractionThread().isRunning();
#endif

  if(!running &&
     (nodes != _isocontouring->isocontourExtractionThread().nodes() ||
      c_nodes != _isocontouring->isocontourExtractionThread().c_nodes()))
    {
      _isocontouring->isocontourExtractionThread().setNodes(nodes,c_nodes);
      _isocontouring->isocontourExtractionThread().start();
    }
#endif

  //now grab the color table, upload it and re-draw
  boost::shared_array<unsigned char> table = _colorTable->getTable();
  _thumbnailViewer->colorTable(table.get());
  _subvolumeViewer->colorTable(table.get());
}

#ifdef THREADED_ISOCONTOURING
IsocontouringManager::IsocontourExtractionThread::IsocontourExtractionThread(VolumeViewerPage *vvp,
									 unsigned int stackSize)
  :
#if QT_VERSION < 0x040000 
  QThread(stackSize), 
#endif
  _volumeViewerPage(vvp)
{
#if QT_VERSION > 0x040000
  setStackSize(stackSize);
#endif
}

void IsocontouringManager::IsocontourExtractionThread::run()
{
  _volumeViewerPage->doGeometryExtraction(_nodes,_c_nodes);
}

void IsocontouringManager::IsocontourExtractionThread::setNodes(const CVCColorTable::isocontour_nodes& n,
							    const CVCColorTable::color_nodes& c)
{
  wait();
  _nodes = n;
  _c_nodes = c;
}

void IsocontouringManager::IsocontourExtractionThread::clearNodes()
{
  wait();
  _nodes.clear();
  _c_nodes.clear();
}

#endif
