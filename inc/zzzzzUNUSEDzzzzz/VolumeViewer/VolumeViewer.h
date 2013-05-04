#ifndef __VOLUMEVIEWER_H__
#define __VOLUMEVIEWER_H__

#include <string.h>
#include <assert.h>
#include <vector>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <QGLViewer/qglviewer.h>
#include <QGLViewer/vec.h>
#include <VolMagick/VolMagick.h>
#include <VolumeLibrary/VolumeRenderer.h>

#include <cvcraw_geometry/cvcraw_geometry.h>
#include <cvcraw_geometry/cvcgeom.h>

namespace VolumeViewer
{
  class Viewer : public QGLViewer
  {
  Q_OBJECT

  public:

    enum VolumeRenderingType { ColorMapped, RGBA };

#if QT_VERSION >= 0x040000
    explicit Viewer(QWidget* parent=0, 
		    const QGLWidget* shareWidget=0, 
		    Qt::WFlags flags=0)
      : QGLViewer(parent,shareWidget,flags) { defaultConstructor(); }
    explicit Viewer(const QGLFormat& format, 
		    QWidget* parent=0, 
		    const QGLWidget* shareWidget=0, 
		    Qt::WFlags flags=0)
      : QGLViewer(format,parent,shareWidget,flags) { defaultConstructor(); }
#endif

#if QT_VERSION < 0x040000 || defined QT3_SUPPORT
    explicit Viewer(QWidget* parent=NULL, 
		       const char* name=0, 
		       const QGLWidget* shareWidget=0, 
		       Qt::WFlags flags=0)
      : QGLViewer(parent, name, shareWidget, flags) { defaultConstructor(); }
      
    explicit Viewer(const QGLFormat& format, 
		       QWidget* parent=0, 
		       const char* name=0, 
		       const QGLWidget* shareWidget=0,
		       Qt::WFlags flags=0)
      : QGLViewer(format, parent, name, shareWidget, flags) { defaultConstructor(); }
#endif

    bool hasBeenInitialized() const { return _hasBeenInitialized; }

    void colorMappedVolume(const VolMagick::Volume& v);
    const VolMagick::Volume& colorMappedVolume() const { return _cmVolume; }
    void rgbaVolumes(const std::vector<VolMagick::Volume>& v);
    const std::vector<VolMagick::Volume>& rgbaVolumes() const { return _rgbaVolumes; }

    //experimental accessors to references...
    VolMagick::Volume& colorMappedVolume()
      {
	_cmVolumeUploaded = false;
	return _cmVolume;
      }

    std::vector<VolMagick::Volume>& rgbaVolumes()
      {
	_rgbaVolumeUploaded = false;
	return _rgbaVolumes;
      }

    bool drawBoundingBox() const { return _drawBoundingBox; }
    VolumeRenderingType volumeRenderingType() const { return _volumeRenderingType; }
    const unsigned char* colorTable() const { return _colorTable; }
    bool drawSubVolumeSelector() const { return _drawSubVolumeSelector;}

    bool drawable() const { return _drawable; }
    bool usingVBO() const { return _usingVBO; }

    VolMagick::BoundingBox cmSubVolumeSelector() const { return _cmSubVolumeSelector; }
    VolMagick::BoundingBox rgbaSubVolumeSelector() const { return _rgbaSubVolumeSelector; }
    VolMagick::BoundingBox subVolumeSelector() const
      {
	switch(volumeRenderingType())
	  {
	  default:
	  case ColorMapped: return cmSubVolumeSelector();
	  case RGBA: return rgbaSubVolumeSelector();
	  }
      }
    VolMagick::BoundingBox boundingBox() const
      {
	switch(volumeRenderingType())
	  {
	  default:
	  case ColorMapped: return _cmVolume.boundingBox();
	  case RGBA: return !_rgbaVolumes.empty() ? 
	      _rgbaVolumes[0].boundingBox() : VolMagick::BoundingBox();
	  }
      }

    bool drawCornerAxis() const { return _drawCornerAxis; }
    bool drawGeometry() const { return _drawGeometry; }
    bool drawVolumes() const { return _drawVolumes; }

    typedef cvcraw_geometry::geometry_t geometry_t;
    struct scene_geometry_t
    {
      enum render_mode_t { POINTS, LINES,
			   TRIANGLES, TRIANGLE_WIREFRAME,
			   TRIANGLE_FILLED_WIRE, QUADS,
			   QUAD_WIREFRAME, QUAD_FILLED_WIRE,
                           TETRA, HEXA };

      geometry_t geometry;
      std::string name;
      render_mode_t render_mode;

      //VBO stuff.. unused if VBO is not supported
      unsigned int vboArrayBufferID;
      unsigned int vboArrayOffsets[3];
      unsigned int vboVertSize;
      unsigned int vboLineElementArrayBufferID, vboLineSize;
      unsigned int vboTriElementArrayBufferID, vboTriSize;
      unsigned int vboQuadElementArrayBufferID, vboQuadSize;

      scene_geometry_t(const geometry_t& geom = geometry_t(),
		       const std::string& n = std::string(),
		       render_mode_t mode = TRIANGLES)
	: geometry(geom), name(n), render_mode(mode) { reset_vbo_info(); }

      scene_geometry_t(const scene_geometry_t& sg)
	: geometry(sg.geometry), name(sg.name), render_mode(sg.render_mode) { reset_vbo_info(); }

      scene_geometry_t& operator=(const scene_geometry_t& geom)
      {
	geometry = geom.geometry;
	name = geom.name;
	render_mode = geom.render_mode;
	reset_vbo_info();
	return *this;
      }

      void reset_vbo_info()
      {
	vboArrayBufferID = 0;
	std::fill(vboArrayOffsets, vboArrayOffsets + 3, 0);
	vboVertSize = 0;
	vboLineElementArrayBufferID = 0;
	vboTriElementArrayBufferID = 0;
	vboQuadElementArrayBufferID = 0;
	vboLineSize = 0;
	vboTriSize = 0;
	vboQuadSize = 0;
      }
    };

    typedef boost::shared_ptr<scene_geometry_t> scene_geometry_ptr;
    typedef std::map<std::string, scene_geometry_ptr> scene_geometry_collection;
    scene_geometry_collection geometries() const;
    scene_geometry_collection& geometries() 
      { 
	_vboUpdated = false;
	return _geometries; 
      }

    void addGeometry(const cvcraw_geometry::geometry_t& geom,
		     const std::string& name = std::string("geometry"),
		     scene_geometry_t::render_mode_t mode = scene_geometry_t::TRIANGLES,
		     bool do_updategl = true)
    {
      //calculate normals if we don't have any
      if(geom.normals.size() != geom.points.size())
	{
	  geometry_t newgeom(geom);
	  newgeom.calculate_surf_normals();
	  scene_geometry_t newscene_geom(newgeom,name,mode);
	  geometries()[name] = scene_geometry_ptr(new scene_geometry_t(newscene_geom));
	}
      else
	{
	  geometries()[name] =
	    scene_geometry_ptr(new scene_geometry_t(geom,name,mode));
	}

      if(do_updategl) updateGL();
    }

    bool clipGeometryToBoundingBox() const { return _clipGeometryToBoundingBox; }

  public slots:
    void drawBoundingBox(bool draw);
    void volumeRenderingType(VolumeRenderingType t);
    void colorTable(const unsigned char * table);
    void drawSubVolumeSelector(bool show);
    void quality(double q);
    void nearPlane(double n);
    void drawCornerAxis(bool draw);
    void drawGeometry(bool draw);
    void drawVolumes(bool draw);
    void clipGeometryToBoundingBox(bool clip);

    void reset(); //resets the viewer to it's default state

  signals:
    void subVolumeSelectorChanged(const VolMagick::BoundingBox& subvolbox);
    void postInit(); //emitted after init() was called

  private:
    void defaultConstructor();

  protected:
    virtual void init();
    virtual void draw();
    virtual void postDraw();
    virtual void drawWithNames();
    virtual void endSelection(const QPoint&);
    virtual void postSelection(const QPoint& point);
    virtual QString helpString() const;

    //all drawing functions must be called AFTER init() has been called...
    void doDrawBoundingBox();
    void doDrawSubVolumeBoundingBox();
    void doDrawSubVolumeSelector(bool withNames = false); //must be called last because of the depth buffer clear
    void doDrawCornerAxis();
    void doDrawGeometry();
    void uploadColorMappedVolume();
    void uploadRGBAVolume();
    void uploadColorTable();
    void updateVBO();
    void setClipPlanes();
    void disableClipPlanes();

    void normalizeScene(); //call to re-center scene around the volume... 
                           //only needed when volume changes, or rendering type changes

    void drawHandle(const qglviewer::Vec& from, const qglviewer::Vec& to, 
		    float radius = -1.0, int nbSubdivisions = 12, 
		    int headName = 0, int shaftName = 0);

    void drawHandle(const qglviewer::Vec& from, const qglviewer::Vec& to,
		    float r = 1.0f, float g = 1.0f, float b = 1.0f,
		    int headName = 0, int shaftName = 0,
		    float pointSize = 2.0f, float lineWidth = 2.0f);
    
    enum HandleName
      {
	MaxXHead = 1, MaxXShaft,
	MinXHead, MinXShaft,
	MaxYHead, MaxYShaft,
	MinYHead, MinYShaft,
	MaxZHead, MaxZShaft,
	MinZHead, MinZShaft,
      };
        
    // Mouse events functions
    virtual void mousePressEvent(QMouseEvent *e);
    virtual void mouseMoveEvent(QMouseEvent *e);
    virtual void mouseReleaseEvent(QMouseEvent *e);

    bool _hasBeenInitialized; //this is set to true once init() has been called.
                              //Do not do any GL calls before this is true!!!!!!!
    
    VolumeRenderer _cmRenderer; //the color mapped volume renderer
    VolumeRenderer _rgbaRenderer; //the RGBA renderer
    VolMagick::Volume _cmVolume; //the color mapped volume (each voxel is unsigned char)
    std::vector<VolMagick::Volume> _rgbaVolumes; //the volumes that make up RGBA components
    unsigned char _colorTable[256*4]; //the color table
    bool _cmVolumeUploaded;
    bool _rgbaVolumeUploaded;

    bool _drawBoundingBox;
    VolumeRenderingType _volumeRenderingType;

    bool _drawable;

    bool _drawSubVolumeSelector;
    VolMagick::BoundingBox _cmSubVolumeSelector;
    VolMagick::BoundingBox _rgbaSubVolumeSelector;

    //the following are used to maintain the selection state
    int _selectedObj;
    QPoint _selectedPoint;

    bool _drawCornerAxis;

    bool _drawGeometry;
    scene_geometry_collection _geometries;

    bool _drawVolumes;

    //If this is true, then we have updated the on-card vertex buffer objects
    //with the contents of _geometries since it was last changed.
    bool _usingVBO;
    bool _vboUpdated;
    //id's of allocated buffers that we should clean up when necessary
    std::vector<unsigned int> _allocatedBuffers;

    //if this is true, geometry is clipped by the bounding box of the current volume
    bool _clipGeometryToBoundingBox;
  };

  typedef Viewer::scene_geometry_ptr scene_geometry_ptr;
  typedef Viewer::scene_geometry_t   scene_geometry_t;
}

#endif
