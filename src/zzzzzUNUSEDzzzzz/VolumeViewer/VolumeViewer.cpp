#include <algorithm>
#include <boost/array.hpp>
#include <boost/scoped_array.hpp>
#include <cmath>
#include <vector>
#include <VolumeViewer/VolumeViewer.h>

#ifdef WIN32
#include <float.h>
#endif

#if QT_VERSION < 0x040000
# include <qapplication.h>
# include <qprogressdialog.h>
#else
# include <QApplication>
# include <QProgressDialog>
# include <QMouseEvent>
#endif

//#define DISABLE_VBO_SUPPORT

namespace VolumeViewer
{
  void Viewer::defaultConstructor()
  {
    //NO GL CALLS ALLOWED HERE!!!! do them in init()

    _hasBeenInitialized = false;

    _drawBoundingBox = false;
    memset(_colorTable,0,sizeof(unsigned char)*256*4);
    _volumeRenderingType = ColorMapped;
    
    //set the color table to greyscale for now
    for(int i=0; i<256; i++)
      {
	_colorTable[i*4+0] = i;
	_colorTable[i*4+1] = i;
	_colorTable[i*4+2] = i;
	_colorTable[i*4+3] = i;
      }
    
    _drawable = false;
    _cmVolumeUploaded = false;
    _rgbaVolumeUploaded = false;
    
    _cmVolume = VolMagick::Volume(VolMagick::Dimension(4,4,4),
				  VolMagick::UChar,
				  VolMagick::BoundingBox(-0.5,-0.5,-0.5,0.5,0.5,0.5));
    _rgbaVolumes = std::vector<VolMagick::Volume>(4,VolMagick::Volume(VolMagick::Dimension(4,4,4),
								      VolMagick::UChar,
								      VolMagick::BoundingBox(-0.5,-0.5,-0.5,
											     0.5,0.5,0.5)));

    _drawSubVolumeSelector = false;
    _cmSubVolumeSelector = VolMagick::BoundingBox(-0.25,-0.25,-0.25,0.25,0.25,0.25);
    _rgbaSubVolumeSelector = VolMagick::BoundingBox(-0.25,-0.25,-0.25,0.25,0.25,0.25);

    _selectedObj = -1;

    _drawCornerAxis = false;

    _drawGeometry = true;
    _drawVolumes = true;
    _usingVBO = false;
    _vboUpdated = false;
    _geometries.clear();
    
    _clipGeometryToBoundingBox = true;
  }

  void Viewer::colorMappedVolume(const VolMagick::Volume& v)
  {
    _cmVolume = v;
    _cmSubVolumeSelector = VolMagick::BoundingBox(v.XMin() + (v.XMax()-v.XMin())/4.0,
						  v.YMin() + (v.YMax()-v.YMin())/4.0,
						  v.ZMin() + (v.ZMax()-v.ZMin())/4.0,
						  v.XMax() - (v.XMax()-v.XMin())/4.0,
						  v.YMax() - (v.YMax()-v.YMin())/4.0,
						  v.ZMax() - (v.ZMax()-v.ZMin())/4.0);

    _cmVolumeUploaded = false;
    normalizeScene();
  }

  void Viewer::rgbaVolumes(const std::vector<VolMagick::Volume>& v)
  {
    assert(v.size() == 4);
    assert(v[0].dimension() == v[1].dimension() &&
	   v[0].dimension() == v[2].dimension() &&
	   v[0].dimension() == v[3].dimension());
    assert(v[0].boundingBox() == v[1].boundingBox() &&
	   v[0].boundingBox() == v[2].boundingBox() &&
	   v[0].boundingBox() == v[3].boundingBox());
    
    _rgbaVolumes = v;
    _rgbaSubVolumeSelector = VolMagick::BoundingBox(v[0].XMin() + (v[0].XMax()-v[0].XMin())/4.0,
						    v[0].YMin() + (v[0].YMax()-v[0].YMin())/4.0,
						    v[0].ZMin() + (v[0].ZMax()-v[0].ZMin())/4.0,
						    v[0].XMax() - (v[0].XMax()-v[0].XMin())/4.0,
						    v[0].YMax() - (v[0].YMax()-v[0].YMin())/4.0,
						    v[0].ZMax() - (v[0].ZMax()-v[0].ZMin())/4.0);
    _rgbaVolumeUploaded = false;
    normalizeScene();
  }

  void Viewer::drawBoundingBox(bool draw)
  {
    _drawBoundingBox = draw;
    if(hasBeenInitialized()) updateGL();
  }

  void Viewer::volumeRenderingType(VolumeRenderingType t)
  {
    makeCurrent();
    _volumeRenderingType = t;
    normalizeScene();
    if(hasBeenInitialized()) updateGL();
  }

  void Viewer::colorTable(const unsigned char *table)
  {
    memcpy(_colorTable, table, sizeof(unsigned char)*256*4);
    uploadColorTable();
    if(hasBeenInitialized()) updateGL();
  }

  void Viewer::drawSubVolumeSelector(bool show)
  {
    _drawSubVolumeSelector = show;
    if(hasBeenInitialized()) updateGL();
  }

  void Viewer::quality(double q)
  {
    makeCurrent();
    _cmRenderer.setQuality(q);
    _rgbaRenderer.setQuality(q);
    if(hasBeenInitialized()) updateGL();
  }

  void Viewer::nearPlane(double n)
  {
    makeCurrent();
    _cmRenderer.setNearPlane(n);
    _rgbaRenderer.setNearPlane(n);
    if(hasBeenInitialized()) updateGL();
  }

  void Viewer::drawCornerAxis(bool draw)
  {
    _drawCornerAxis = draw;
    if(hasBeenInitialized()) updateGL();
  }

  void Viewer::drawGeometry(bool draw)
  {
    _drawGeometry = draw;
    if(hasBeenInitialized()) updateGL();
  }

  void Viewer::drawVolumes(bool draw)
  {
    _drawVolumes = draw;
    if(hasBeenInitialized()) updateGL();
  }

  void Viewer::clipGeometryToBoundingBox(bool draw)
  {
    _clipGeometryToBoundingBox = draw;
    if(hasBeenInitialized()) updateGL();
  }

  Viewer::scene_geometry_collection Viewer::geometries() const
  {
    using namespace std;
    scene_geometry_collection copy(_geometries);
    for(scene_geometry_collection::iterator i = copy.begin();
	i != copy.end();
	i++)
      i->second.reset(new scene_geometry_t(i->second->geometry,
					   i->second->name,
					   i->second->render_mode));
    return copy;
  }

  void Viewer::reset()
  {
    makeCurrent();
    defaultConstructor();
    init();
    if(hasBeenInitialized()) updateGL();
  }

  void Viewer::init()
  {
    setMouseTracking(true);

    //initialize glew before initializing the renderers!!
    GLenum err = glewInit();
    if(GLEW_OK != err)
      qDebug("Error: %s", glewGetErrorString(err));
    qDebug("Status: Using GLEW %s",glewGetString(GLEW_VERSION));
    
    if(!_cmRenderer.initRenderer() ||
       !_rgbaRenderer.initRenderer())
      _drawable = false;
    else
      {
	uploadColorTable();
	normalizeScene();
      }

    emit postInit();
    _hasBeenInitialized = true;
  }
  
  void Viewer::draw()
  {
    if(drawBoundingBox()) doDrawBoundingBox();
    if(drawSubVolumeSelector()) doDrawSubVolumeBoundingBox();
    if(drawGeometry()) doDrawGeometry();

    if(drawVolumes())
      {
        //Something to note about the volume renderer: it always draws volumes with the center of the volume
        //at the origin.  So to correctly position the volume in the scene, we must translate and scale it
        //so it fits its designated bounding box.
        glPushMatrix();
        glTranslatef(sceneCenter().x,sceneCenter().y,sceneCenter().z);
        switch(volumeRenderingType())
          {
          case ColorMapped:
            {
              if(!_cmVolumeUploaded)
                uploadColorMappedVolume();
	  
              double maxd = std::max(_cmVolume.XMax()-_cmVolume.XMin(),
                                     std::max(_cmVolume.YMax()-_cmVolume.YMin(),
                                              _cmVolume.ZMax()-_cmVolume.ZMin()));
              double aspectx = (_cmVolume.XMax()-_cmVolume.XMin())/maxd;
              double aspecty = (_cmVolume.YMax()-_cmVolume.YMin())/maxd;
              double aspectz = (_cmVolume.ZMax()-_cmVolume.ZMin())/maxd;

              glScalef((_cmVolume.XMax()-_cmVolume.XMin())/aspectx,
                       (_cmVolume.YMax()-_cmVolume.YMin())/aspecty,
                       (_cmVolume.ZMax()-_cmVolume.ZMin())/aspectz);

              _cmRenderer.renderVolume();
            }
            break;
          case RGBA:
            {
              if(!_rgbaVolumeUploaded)
                uploadRGBAVolume();
	  
              double maxd = std::max(_rgbaVolumes[0].XMax()-_rgbaVolumes[0].XMin(),
                                     std::max(_rgbaVolumes[0].YMax()-_rgbaVolumes[0].YMin(),
                                              _rgbaVolumes[0].ZMax()-_rgbaVolumes[0].ZMin()));
              double aspectx = (_rgbaVolumes[0].XMax()-_rgbaVolumes[0].XMin())/maxd;
              double aspecty = (_rgbaVolumes[0].YMax()-_rgbaVolumes[0].YMin())/maxd;
              double aspectz = (_rgbaVolumes[0].ZMax()-_rgbaVolumes[0].ZMin())/maxd;

              glScalef((_rgbaVolumes[0].XMax()-_rgbaVolumes[0].XMin())/aspectx,
                       (_rgbaVolumes[0].YMax()-_rgbaVolumes[0].YMin())/aspecty,
                       (_rgbaVolumes[0].ZMax()-_rgbaVolumes[0].ZMin())/aspectz);

              _rgbaRenderer.renderVolume();
            }
            break;
          default: break;
          }
        glPopMatrix();
      }

    if(drawSubVolumeSelector()) doDrawSubVolumeSelector();
  }

  void Viewer::drawWithNames()
  {
    if(drawSubVolumeSelector()) doDrawSubVolumeSelector(true);
  }

  void Viewer::postDraw()
  {
    QGLViewer::postDraw();
    if(drawCornerAxis()) doDrawCornerAxis();
  }

  void Viewer::endSelection(const QPoint&)
  {
    glFlush();

    GLint nbHits = glRenderMode(GL_RENDER);
    
    if(nbHits <= 0)
      setSelectedName(-1);
    else
      {
	for(int i=0; i<nbHits; i++)
	  {
	    //prefer to select the heads over shafts
	    switch((selectBuffer())[i*4+3])
	      {
	      case MaxXHead:
	      case MinXHead:
	      case MaxYHead:
	      case MinYHead:
	      case MaxZHead:
	      case MinZHead:
		setSelectedName((selectBuffer())[i*4+3]);
		return;
	      }
	       
	    setSelectedName((selectBuffer())[i*4+3]);
	  }
      }
  }

  void Viewer::postSelection(const QPoint& point)
  {
    //qDebug("Selected: %d", selectedName());
    _selectedPoint = point;
    _selectedObj = selectedName();
  }

  QString Viewer::helpString() const
  {
    return QString("<h1>Nothing here right now...</h1>");
  }

  void Viewer::doDrawBoundingBox()
  {
    double minx, miny, minz;
    double maxx, maxy, maxz;
    float bgcolor[4];

    switch(volumeRenderingType())
      {
      case RGBA:
	minx = _rgbaVolumes[0].XMin();
	miny = _rgbaVolumes[0].YMin();
	minz = _rgbaVolumes[0].ZMin();
	maxx = _rgbaVolumes[0].XMax();
	maxy = _rgbaVolumes[0].YMax();
	maxz = _rgbaVolumes[0].ZMax();
	break;
      default:;
      case ColorMapped:
	minx = _cmVolume.XMin();
	miny = _cmVolume.YMin();
	minz = _cmVolume.ZMin();
	maxx = _cmVolume.XMax();
	maxy = _cmVolume.YMax();
	maxz = _cmVolume.ZMax();
	break;
      }

    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT);
    glLineWidth(1.0);
    glEnable(GL_LINE_SMOOTH);
    glDisable(GL_LIGHTING);
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgcolor);
    glColor4f(1.0-bgcolor[0],1.0-bgcolor[1],1.0-bgcolor[2],1.0-bgcolor[3]);
    /* front face */
    glBegin(GL_LINE_LOOP);
    glVertex3d(minx,miny,minz);
    glVertex3d(maxx,miny,minz);
    glVertex3d(maxx,maxy,minz);
    glVertex3d(minx,maxy,minz);
    glEnd();
    /* back face */
    glBegin(GL_LINE_LOOP);
    glVertex3d(minx,miny,maxz);
    glVertex3d(maxx,miny,maxz);
    glVertex3d(maxx,maxy,maxz);
    glVertex3d(minx,maxy,maxz);
    glEnd();
    /* connecting lines */
    glBegin(GL_LINES);
    glVertex3d(minx,maxy,minz);
    glVertex3d(minx,maxy,maxz);
    glVertex3d(minx,miny,minz);
    glVertex3d(minx,miny,maxz);
    glVertex3d(maxx,maxy,minz);
    glVertex3d(maxx,maxy,maxz);
    glVertex3d(maxx,miny,minz);
    glVertex3d(maxx,miny,maxz);
    glEnd();
    glDisable(GL_LINE_SMOOTH);
    glPopAttrib();
  }

  void Viewer::doDrawSubVolumeBoundingBox()
  {
    double minx, miny, minz;
    double maxx, maxy, maxz;
    float bgcolor[4];
    
    switch(volumeRenderingType())
      {
      case RGBA:
	minx = _rgbaSubVolumeSelector.minx;
	miny = _rgbaSubVolumeSelector.miny;
	minz = _rgbaSubVolumeSelector.minz;
	maxx = _rgbaSubVolumeSelector.maxx;
	maxy = _rgbaSubVolumeSelector.maxy;
	maxz = _rgbaSubVolumeSelector.maxz;

	break;
      default:;
      case ColorMapped:
	minx = _cmSubVolumeSelector.minx;
	miny = _cmSubVolumeSelector.miny;
	minz = _cmSubVolumeSelector.minz;
	maxx = _cmSubVolumeSelector.maxx;
	maxy = _cmSubVolumeSelector.maxy;
	maxz = _cmSubVolumeSelector.maxz;

	break;
      }

    glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT);
    glLineWidth(1.0);
    glEnable(GL_LINE_SMOOTH);
    glDisable(GL_LIGHTING);
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgcolor);
    glColor4f(1.0-bgcolor[0],1.0-bgcolor[1],1.0-bgcolor[2],1.0-bgcolor[3]);
    /* front face */
    glBegin(GL_LINE_LOOP);
    glVertex3d(minx,miny,minz);
    glVertex3d(maxx,miny,minz);
    glVertex3d(maxx,maxy,minz);
    glVertex3d(minx,maxy,minz);
    glEnd();
    /* back face */
    glBegin(GL_LINE_LOOP);
    glVertex3d(minx,miny,maxz);
    glVertex3d(maxx,miny,maxz);
    glVertex3d(maxx,maxy,maxz);
    glVertex3d(minx,maxy,maxz);
    glEnd();
    /* connecting lines */
    glBegin(GL_LINES);
    glVertex3d(minx,maxy,minz);
    glVertex3d(minx,maxy,maxz);
    glVertex3d(minx,miny,minz);
    glVertex3d(minx,miny,maxz);
    glVertex3d(maxx,maxy,minz);
    glVertex3d(maxx,maxy,maxz);
    glVertex3d(maxx,miny,minz);
    glVertex3d(maxx,miny,maxz);
    glEnd();
    glDisable(GL_LINE_SMOOTH);
    glPopAttrib();
  }

  void Viewer::doDrawSubVolumeSelector(bool withNames)
  {
    using namespace qglviewer;

    double minx, miny, minz;
    double maxx, maxy, maxz;
    
    switch(volumeRenderingType())
      {
      case RGBA:
	minx = _rgbaSubVolumeSelector.minx;
	miny = _rgbaSubVolumeSelector.miny;
	minz = _rgbaSubVolumeSelector.minz;
	maxx = _rgbaSubVolumeSelector.maxx;
	maxy = _rgbaSubVolumeSelector.maxy;
	maxz = _rgbaSubVolumeSelector.maxz;

	break;
      default:;
      case ColorMapped:
	minx = _cmSubVolumeSelector.minx;
	miny = _cmSubVolumeSelector.miny;
	minz = _cmSubVolumeSelector.minz;
	maxx = _cmSubVolumeSelector.maxx;
	maxy = _cmSubVolumeSelector.maxy;
	maxz = _cmSubVolumeSelector.maxz;

	break;
      }

    double centerx, centery, centerz;
    centerx = (maxx-minx)/2+minx;
    centery = (maxy-miny)/2+miny;
    centerz = (maxz-minz)/2+minz;

    /*** draw the selector handle arrows ***/
    float pointSize = withNames ? 6.0 : 6.0;
    float lineWidth = withNames ? 4.0 : 2.0;

    glClear(GL_DEPTH_BUFFER_BIT); //show arrows on top of everything
    drawHandle(Vec(centerx,centery,centerz),
	       Vec(maxx,centery,centerz),
	       1.0,0.0,0.0,
	       MaxXHead,MaxXShaft,
	       pointSize,lineWidth);
    drawHandle(Vec(centerx,centery,centerz),
	       Vec(minx,centery,centerz),
	       1.0,0.0,0.0,
	       MinXHead,MinXShaft,
	       pointSize,lineWidth);

    drawHandle(Vec(centerx,centery,centerz),
	       Vec(centerx,maxy,centerz),
	       0.0,1.0,0.0,
	       MaxYHead,MaxYShaft,
	       pointSize,lineWidth);
    drawHandle(Vec(centerx,centery,centerz),
	       Vec(centerx,miny,centerz),
	       0.0,1.0,0.0,
	       MinYHead,MinYShaft,
	       pointSize,lineWidth);
    
    drawHandle(Vec(centerx,centery,centerz),
	       Vec(centerx,centery,maxz),
	       0.0,0.0,1.0,
	       MaxZHead,MaxZShaft,
	       pointSize,lineWidth);
    drawHandle(Vec(centerx,centery,centerz),
	       Vec(centerx,centery,minz),
	       0.0,0.0,1.0,
	       MinZHead,MinZShaft,
	       pointSize,lineWidth);
  }

  void Viewer::doDrawCornerAxis()
  {
    GLint viewport[4];
    GLint scissor[4];

    // The viewport and the scissor are changed to fit the lower left
    // corner. Original values are saved.
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetIntegerv(GL_SCISSOR_BOX, scissor);

    // Axis viewport size, in pixels
    const GLint size = 50; 
    glViewport(0,0,size,size);
    glScissor(0,0,size,size);

    // The Z-buffer is cleared to make the axis appear over the
    // original image.
    glClear(GL_DEPTH_BUFFER_BIT);

    // Tune for best line rendering
    glDisable(GL_LIGHTING);
    glLineWidth(1.0); 

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMultMatrixd(camera()->orientation().inverse().matrix());

    glBegin(GL_LINES);
    glColor3f(1.0, 0.0, 0.0);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(1.0, 0.0, 0.0);

    glColor3f(0.0, 1.0, 0.0);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 1.0, 0.0);
	
    glColor3f(0.0, 0.0, 1.0);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 0.0, 1.0);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glEnable(GL_LIGHTING);

    // The viewport and the scissor are restored.
    glScissor(scissor[0],scissor[1],scissor[2],scissor[3]);
    glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
  }

  void Viewer::doDrawGeometry()
  {
    if(!_vboUpdated) updateVBO();

    glPushAttrib(GL_LIGHTING_BIT | GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glEnable(GL_NORMALIZE);

    if(clipGeometryToBoundingBox())
      setClipPlanes();

    for(scene_geometry_collection::iterator i = _geometries.begin();
	i != _geometries.end();
	i++)
      {
	if(!i->second) continue;

	scene_geometry_t &scene_geom = *(i->second.get());
	geometry_t &geom = scene_geom.geometry;

	GLint params[2];
	//back up current setting
	glGetIntegerv(GL_POLYGON_MODE,params);

	glEnable(GL_LIGHTING);

	//make sure we have normals!
	if(geom.points.size() != geom.normals.size())
	  geom.calculate_surf_normals();

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	if(usingVBO())
	  {
	    glBindBufferARB(GL_ARRAY_BUFFER_ARB,scene_geom.vboArrayBufferID);
	    glVertexPointer(3, GL_DOUBLE, 0, (GLvoid*)scene_geom.vboArrayOffsets[0]);
	    glNormalPointer(GL_DOUBLE, 0, (GLvoid*)scene_geom.vboArrayOffsets[1]);
	  }
	else
	  {
	    glVertexPointer(3, GL_DOUBLE, 0, &(geom.points[0]));
	    glNormalPointer(GL_DOUBLE, 0, &(geom.normals[0]));
	  }
	    
	if(geom.colors.size() == geom.points.size())
	  {
	    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
	    glEnable(GL_COLOR_MATERIAL);
	    glEnableClientState(GL_COLOR_ARRAY);
	    if(usingVBO())
	      glColorPointer(3, GL_DOUBLE, 0, (GLvoid*)scene_geom.vboArrayOffsets[2]);
	    else
	      glColorPointer(3, GL_DOUBLE, 0, &(geom.colors[0]));
	  }
	else
	  glColor3f(1.0,1.0,1.0);
	   
	switch(i->second->render_mode)
	  {
	  case scene_geometry_t::POINTS:
	    {
	      glPushAttrib(GL_LIGHTING_BIT | GL_POINT_BIT);
	      glDisable(GL_LIGHTING);
	      glPointSize(2.0);
	      glEnable(GL_POINT_SMOOTH);

	      //no need for normals when point rendering
	      glDisableClientState(GL_NORMAL_ARRAY);

	      if(usingVBO())
		glDrawArrays(GL_POINTS, 0, scene_geom.vboVertSize/3);
	      else
		glDrawArrays(GL_POINTS, 0, geom.points.size());

	      glPopAttrib();
	    }
	    break;
	  case scene_geometry_t::LINES:
	    if(usingVBO())
	      {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
				scene_geom.vboLineElementArrayBufferID);
		glDrawElements(GL_LINES, 
			       geom.lines.size()*2,
			       GL_UNSIGNED_INT, 0);
	      }
	    else
	      glDrawElements(GL_LINES, 
			     geom.lines.size()*2,
			     GL_UNSIGNED_INT, &(geom.lines[0]));
	    break;
	  case scene_geometry_t::TRIANGLES:
	    if(usingVBO())
	      {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
				scene_geom.vboTriElementArrayBufferID);
		glDrawElements(GL_TRIANGLES, 
			       geom.tris.size()*3,
			       GL_UNSIGNED_INT, 0);
	      }
	    else
	      glDrawElements(GL_TRIANGLES, 
			     geom.tris.size()*3,
			     GL_UNSIGNED_INT, &(geom.tris[0]));
	    break;
	  case scene_geometry_t::TRIANGLE_WIREFRAME:
	    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	    if(usingVBO())
	      {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
				scene_geom.vboTriElementArrayBufferID);
		glDrawElements(GL_TRIANGLES, 
			       geom.tris.size()*3,
			       GL_UNSIGNED_INT, 0);
	      }
	    else
	      glDrawElements(GL_TRIANGLES, 
			     geom.tris.size()*3,
			     GL_UNSIGNED_INT, &(geom.tris[0]));
	    break;
	  case scene_geometry_t::TRIANGLE_FILLED_WIRE:
	    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	    glEnable(GL_POLYGON_OFFSET_FILL);
	    glPolygonOffset(1.0,1.0);
	    if(usingVBO())
	      {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
				scene_geom.vboTriElementArrayBufferID);
		glDrawElements(GL_TRIANGLES, 
			       geom.tris.size()*3,
			       GL_UNSIGNED_INT, 0);
	      }
	    else
	      glDrawElements(GL_TRIANGLES, 
			     geom.tris.size()*3,
			     GL_UNSIGNED_INT, &(geom.tris[0]));
	    glPolygonOffset(0.0,0.0);
	    glDisable(GL_POLYGON_OFFSET_FILL);
	    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	    glDisable(GL_LIGHTING);
	    glDisableClientState(GL_COLOR_ARRAY);
	    glColor3f(0.0,0.0,0.0); //black wireframe
	    if(usingVBO())
	      {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
				scene_geom.vboTriElementArrayBufferID);
		glDrawElements(GL_TRIANGLES, 
			       geom.tris.size()*3,
			       GL_UNSIGNED_INT, 0);
	      }
	    else
	      glDrawElements(GL_TRIANGLES, 
			     geom.tris.size()*3,
			     GL_UNSIGNED_INT, &(geom.tris[0]));
	    break;
	  case scene_geometry_t::QUADS:
	    if(usingVBO())
	      {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
				scene_geom.vboQuadElementArrayBufferID);
		glDrawElements(GL_QUADS, 
			       geom.quads.size()*4,
			       GL_UNSIGNED_INT, 0);
	      }
	    else
	      glDrawElements(GL_QUADS, 
			     geom.quads.size()*4,
			     GL_UNSIGNED_INT, &(geom.quads[0]));
	    break;
	  case scene_geometry_t::QUAD_WIREFRAME:
	    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	    if(usingVBO())
	      {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
				scene_geom.vboQuadElementArrayBufferID);
		glDrawElements(GL_QUADS, 
			       geom.quads.size()*4,
			       GL_UNSIGNED_INT, 0);
	      }
	    else
	      glDrawElements(GL_QUADS, 
			     geom.quads.size()*4,
			     GL_UNSIGNED_INT, &(geom.quads[0]));
	    break;
	  case scene_geometry_t::QUAD_FILLED_WIRE:
	    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	    glEnable(GL_POLYGON_OFFSET_FILL);
	    glPolygonOffset(1.0,1.0);
	    if(usingVBO())
	      {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
				scene_geom.vboQuadElementArrayBufferID);
		glDrawElements(GL_QUADS, 
			       geom.quads.size()*4,
			       GL_UNSIGNED_INT, 0);
	      }
	    else
	      glDrawElements(GL_QUADS, 
			     geom.quads.size()*4,
			     GL_UNSIGNED_INT, &(geom.quads[0]));
	    glPolygonOffset(0.0,0.0);
	    glDisable(GL_POLYGON_OFFSET_FILL);
	    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	    glDisable(GL_LIGHTING);
	    glDisableClientState(GL_COLOR_ARRAY);
	    glColor3f(0.0,0.0,0.0); //black wireframe
	    if(usingVBO())
	      {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
				scene_geom.vboQuadElementArrayBufferID);
		glDrawElements(GL_QUADS, 
			       geom.quads.size()*4,
			       GL_UNSIGNED_INT, 0);
	      }
	    else
	      glDrawElements(GL_QUADS, 
			     geom.quads.size()*4,
			     GL_UNSIGNED_INT, &(geom.quads[0]));
	    break;

	  case scene_geometry_t::TETRA: //TRIANGLES and LINES combined

	    if(usingVBO())
	      {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
				scene_geom.vboTriElementArrayBufferID);
		glDrawElements(GL_TRIANGLES, 
			       geom.tris.size()*3,
			       GL_UNSIGNED_INT, 0);
	      }
	    else
	      glDrawElements(GL_TRIANGLES,
			     geom.tris.size()*3,
			     GL_UNSIGNED_INT, &(geom.tris[0]));

	    if(usingVBO())
	      {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
				scene_geom.vboLineElementArrayBufferID);
		glDrawElements(GL_LINES, 
			       geom.lines.size()*2,
			       GL_UNSIGNED_INT, 0);
	      }
	    else
	      glDrawElements(GL_LINES, 
			     geom.lines.size()*2,
			     GL_UNSIGNED_INT, &(geom.lines[0]));

	    break;

	  case scene_geometry_t::HEXA: //QUADS and LINES combined

	    if(usingVBO())
	      {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
				scene_geom.vboQuadElementArrayBufferID);
		glDrawElements(GL_QUADS, 
			       geom.quads.size()*4,
			       GL_UNSIGNED_INT, 0);
	      }
	    else
	      glDrawElements(GL_QUADS, 
			     geom.quads.size()*4,
			     GL_UNSIGNED_INT, &(geom.quads[0]));

	    if(usingVBO())
	      {
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
				scene_geom.vboLineElementArrayBufferID);
		glDrawElements(GL_LINES, 
			       geom.lines.size()*2,
			       GL_UNSIGNED_INT, 0);
	      }
	    else
	      glDrawElements(GL_LINES, 
			     geom.lines.size()*2,
			     GL_UNSIGNED_INT, &(geom.lines[0]));
	  }

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	if(usingVBO())
	  {
	    glBindBufferARB(GL_ARRAY_BUFFER_ARB,0);
	    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,0);
	  }

	//restore previous setting for polygon mode
	glPolygonMode(GL_FRONT,params[0]);
	glPolygonMode(GL_BACK,params[1]);
      }
      
    if(clipGeometryToBoundingBox())
      disableClipPlanes();

    glPopAttrib();
  }

  static inline unsigned int upToPowerOfTwo(unsigned int value)
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

  inline double texCoordOfSample(double sample, 
				 int bufferWidth, 
				 int canvasWidth,
				 double bufferMin, 
				 double bufferMax)
  {
    // get buffer min and max in the texture's space
    double texBufferMin = 0.5 / (double)canvasWidth;
    double texBufferMax = ((double)bufferWidth - 0.5) / (double)canvasWidth;
    
    return (sample-bufferMin)/(bufferMax-bufferMin) * (texBufferMax-texBufferMin) + texBufferMin;
  }

  void Viewer::uploadColorMappedVolume()
  {
    makeCurrent();

    if(_cmVolume.voxelType() != VolMagick::UChar)
      {
	_cmVolume.map(0.0,255.0);
	_cmVolume.voxelType(VolMagick::UChar);
      }
    
    VolMagick::uint64 upload_dimx = upToPowerOfTwo(_cmVolume.XDim());
    VolMagick::uint64 upload_dimy = upToPowerOfTwo(_cmVolume.YDim());
    VolMagick::uint64 upload_dimz = upToPowerOfTwo(_cmVolume.ZDim());
    boost::scoped_array<unsigned char> upload_buf(new unsigned char[upload_dimx*upload_dimy*upload_dimz]);
    for(VolMagick::uint64 k = 0; k < _cmVolume.ZDim(); k++)
      for(VolMagick::uint64 j = 0; j < _cmVolume.YDim(); j++)
	for(VolMagick::uint64 i = 0; i < _cmVolume.XDim(); i++)
	  upload_buf[k*upload_dimx*upload_dimy+j*upload_dimx+i] = 
	    static_cast<unsigned char>(_cmVolume(i,j,k));

    _cmRenderer.uploadColorMappedData(upload_buf.get(),upload_dimx,upload_dimy,upload_dimz);
    _cmRenderer.setAspectRatio(fabs(_cmVolume.XMax()-_cmVolume.XMin()),
			       fabs(_cmVolume.YMax()-_cmVolume.YMin()),
			       fabs(_cmVolume.ZMax()-_cmVolume.ZMin()));
    _cmRenderer.
      setTextureSubCube(texCoordOfSample(_cmVolume.XMin(),
					 _cmVolume.XDim(),
					 upload_dimx,
					 _cmVolume.XMin(),
					 _cmVolume.XMax()),
			texCoordOfSample(_cmVolume.YMin(),
					 _cmVolume.YDim(),
					 upload_dimy,
					 _cmVolume.YMin(),
					 _cmVolume.YMax()),
			texCoordOfSample(_cmVolume.ZMin(),
					 _cmVolume.ZDim(),
					 upload_dimz,
					 _cmVolume.ZMin(),
					 _cmVolume.ZMax()),
			texCoordOfSample(_cmVolume.XMax(),
					 _cmVolume.XDim(),
					 upload_dimx,
					 _cmVolume.XMin(),
					 _cmVolume.XMax()),
			texCoordOfSample(_cmVolume.YMax(),
					 _cmVolume.YDim(),
					 upload_dimy,
					 _cmVolume.YMin(),
					 _cmVolume.YMax()),
			texCoordOfSample(_cmVolume.ZMax(),
					 _cmVolume.ZDim(),
					 upload_dimz,
					 _cmVolume.ZMin(),
					 _cmVolume.ZMax()));

    _cmVolumeUploaded = true;
  }

  void Viewer::uploadRGBAVolume()
  {
    makeCurrent();

    for(unsigned int i = 0; i<4; i++)
      {
	if(_rgbaVolumes[i].voxelType() != VolMagick::UChar)
	  {
	    _rgbaVolumes[i].map(0.0,255.0);
	    _rgbaVolumes[i].voxelType(VolMagick::UChar);
	  }
      }

    VolMagick::uint64 upload_dimx = upToPowerOfTwo(_rgbaVolumes[0].XDim());
    VolMagick::uint64 upload_dimy = upToPowerOfTwo(_rgbaVolumes[0].YDim());
    VolMagick::uint64 upload_dimz = upToPowerOfTwo(_rgbaVolumes[0].ZDim());
    boost::scoped_array<unsigned char> upload_buf(new unsigned char[upload_dimx*upload_dimy*upload_dimz*4]);
#if QT_VERSION < 0x040000
    QProgressDialog progress("Creating interleaved RGBA 3D texture...","Abort",_rgbaVolumes[0].ZDim(),NULL,NULL,true);
#else
    QProgressDialog progress("Creating interleaved RGBA 3D texture...","Abort",0,_rgbaVolumes[0].ZDim());
    progress.setWindowModality(Qt::WindowModal);
#endif
    for(VolMagick::uint64 k = 0; k < _rgbaVolumes[0].ZDim(); k++)
      {
#if QT_VERSION < 0x040000
	progress.setProgress(k);
#else
	progress.setValue(k);
#endif
	qApp->processEvents();
	for(VolMagick::uint64 j = 0; j < _rgbaVolumes[0].YDim(); j++)
	  for(VolMagick::uint64 i = 0; i < _rgbaVolumes[0].XDim(); i++)
	    for(int cols = 0; cols < 4; cols++)
	      upload_buf[k*upload_dimx*upload_dimy*4+
			 j*upload_dimx*4+
			 i*4+
			 cols] = 
		static_cast<unsigned char>(_rgbaVolumes[cols](i,j,k));
      }
#if QT_VERSION < 0x040000
    progress.setProgress(_rgbaVolumes[0].ZDim());
#else
    progress.setValue(_rgbaVolumes[0].ZDim());
#endif
    
    _rgbaRenderer.uploadRGBAData(upload_buf.get(),upload_dimx,upload_dimy,upload_dimz);
    _rgbaRenderer.setAspectRatio(fabs(_rgbaVolumes[0].XMax()-_rgbaVolumes[0].XMin()),
				 fabs(_rgbaVolumes[0].YMax()-_rgbaVolumes[0].YMin()),
				 fabs(_rgbaVolumes[0].ZMax()-_rgbaVolumes[0].ZMin()));
    _rgbaRenderer.
      setTextureSubCube(texCoordOfSample(_rgbaVolumes[0].XMin(),
					 _rgbaVolumes[0].XDim(),
					 upload_dimx,
					 _rgbaVolumes[0].XMin(),
					 _rgbaVolumes[0].XMax()),
			texCoordOfSample(_rgbaVolumes[0].YMin(),
					 _rgbaVolumes[0].YDim(),
					 upload_dimy,
					 _rgbaVolumes[0].YMin(),
					 _rgbaVolumes[0].YMax()),
			texCoordOfSample(_rgbaVolumes[0].ZMin(),
					 _rgbaVolumes[0].ZDim(),
					 upload_dimz,
					 _rgbaVolumes[0].ZMin(),
					 _rgbaVolumes[0].ZMax()),
			texCoordOfSample(_rgbaVolumes[0].XMax(),
					 _rgbaVolumes[0].XDim(),
					 upload_dimx,
					 _rgbaVolumes[0].XMin(),
					 _rgbaVolumes[0].XMax()),
			texCoordOfSample(_rgbaVolumes[0].YMax(),
					 _rgbaVolumes[0].YDim(),
					 upload_dimy,
					 _rgbaVolumes[0].YMin(),
					 _rgbaVolumes[0].YMax()),
			texCoordOfSample(_rgbaVolumes[0].ZMax(),
					 _rgbaVolumes[0].ZDim(),
					 upload_dimz,
					 _rgbaVolumes[0].ZMin(),
					 _rgbaVolumes[0].ZMax()));

    _rgbaVolumeUploaded = true;
  }

  void Viewer::uploadColorTable()
  {
    makeCurrent();
    _cmRenderer.uploadColorMap(_colorTable);
  }

  void Viewer::updateVBO()
  {
    using std::vector;

    //If this version of opengl doesn't support VBOs, don't do anything.
    if(!glewIsSupported("GL_ARB_vertex_buffer_object") 
#ifdef DISABLE_VBO_SUPPORT
       || true
#endif
       )
      {
	static bool already_printed = false;
	
	if(!already_printed)
	  {
#ifdef DISABLE_VBO_SUPPORT
	    qDebug("Viewer::updateVBO(): VBO support disabled...");
#else
	    qDebug("Viewer::updateVBO(): no VBO support detected...");
#endif
	    already_printed = true;
	  }

	_vboUpdated = true;
	_usingVBO = false;
	return;
      }

    //clean up all previously allocated buffers
    for(vector<unsigned int>::iterator i = _allocatedBuffers.begin();
	i != _allocatedBuffers.end();
	i++)
      if(glIsBufferARB(*i))
	glDeleteBuffersARB(1,reinterpret_cast<GLuint*>(&(*i)));
    _allocatedBuffers.clear();

    //qDebug("Viewer::updateVBO(): generating buffers...");

    for(scene_geometry_collection::iterator i = _geometries.begin();
	i != _geometries.end();
	i++)
      {
	if(!i->second) continue;

	scene_geometry_t &scene_geom = *(i->second.get());
	geometry_t &geom = scene_geom.geometry;
	
	//make sure we have normals!
	if(geom.points.size() != geom.normals.size())
	  geom.calculate_surf_normals();
	
	//get new buffer ids
	glGenBuffersARB(1,reinterpret_cast<GLuint*>(&scene_geom.vboArrayBufferID));
	glGenBuffersARB(1,reinterpret_cast<GLuint*>(&scene_geom.vboLineElementArrayBufferID));
	glGenBuffersARB(1,reinterpret_cast<GLuint*>(&scene_geom.vboTriElementArrayBufferID));
	glGenBuffersARB(1,reinterpret_cast<GLuint*>(&scene_geom.vboQuadElementArrayBufferID));
	_allocatedBuffers.push_back(scene_geom.vboArrayBufferID);
	_allocatedBuffers.push_back(scene_geom.vboLineElementArrayBufferID);
	_allocatedBuffers.push_back(scene_geom.vboTriElementArrayBufferID);
	_allocatedBuffers.push_back(scene_geom.vboQuadElementArrayBufferID);
	
	unsigned int bufsize =
	  geom.points.size()*3 +
	  geom.normals.size()*3 +
	  geom.colors.size()*3;
	boost::scoped_array<double>
	  buf(new double[bufsize]);
	  
	for(unsigned int i = 0; i < geom.points.size(); i++)
	  for(unsigned int j = 0; j < 3; j++)
	    buf[i*3+j] = geom.points[i][j];
	  
	for(unsigned int i = 0, buf_i = geom.points.size();
	    i < geom.normals.size(); 
	    i++, buf_i++)
	  for(unsigned int j = 0; j < 3; j++)
	    buf[buf_i*3+j] = geom.normals[i][j];
	  
	for(unsigned int i = 0, buf_i = geom.points.size()+geom.normals.size();
	    i < geom.colors.size(); 
	    i++, buf_i++)
	  for(unsigned int j = 0; j < 3; j++)
	    buf[buf_i*3+j] = geom.colors[i][j];
	  
	scene_geom.vboVertSize = geom.points.size()*3;
	scene_geom.vboArrayOffsets[0] = 0;
	scene_geom.vboArrayOffsets[1] = geom.points.size()*3*sizeof(double);
	scene_geom.vboArrayOffsets[2] = (geom.points.size()+geom.normals.size())*3*sizeof(double);
	  
	//upload vertex info
	glBindBufferARB(GL_ARRAY_BUFFER_ARB,scene_geom.vboArrayBufferID);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB,
			bufsize*sizeof(double),
			buf.get(),
			GL_STATIC_DRAW_ARB);
	  
	//upload element info
	bufsize = geom.lines.size()*2;
	boost::scoped_array<int> element_buf(new int[bufsize]);
	for(unsigned int i = 0; i < geom.lines.size(); i++)
	  for(unsigned int j = 0; j < 2; j++)
	    element_buf[i*2+j] = geom.lines[i][j];
	  
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,scene_geom.vboLineElementArrayBufferID);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
			bufsize*sizeof(int),
			element_buf.get(),
			GL_STATIC_DRAW_ARB);
	scene_geom.vboLineSize = bufsize;
	  
	bufsize = geom.tris.size()*3;
	element_buf.reset(new int[bufsize]);
	for(unsigned int i = 0; i < geom.tris.size(); i++)
	  for(unsigned int j = 0; j < 3; j++)
	    element_buf[i*3+j] = geom.tris[i][j];
	  
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,scene_geom.vboTriElementArrayBufferID);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
			bufsize*sizeof(int),
			element_buf.get(),
			GL_STATIC_DRAW_ARB);
	scene_geom.vboTriSize = bufsize;
	  
	bufsize = geom.quads.size()*4;
	element_buf.reset(new int[bufsize]);
	for(unsigned int i = 0; i < geom.quads.size(); i++)
	  for(unsigned int j = 0; j < 4; j++)
	    element_buf[i*4+j] = geom.quads[i][j];
	  
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB,scene_geom.vboQuadElementArrayBufferID);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB,
			bufsize*sizeof(int),
			element_buf.get(),
			GL_STATIC_DRAW_ARB);
	scene_geom.vboQuadSize = bufsize;
      }
    
    _usingVBO = true;
    _vboUpdated = true;
  }

  void Viewer::setClipPlanes()
  {
    VolMagick::BoundingBox bbox = boundingBox();
    
    double plane0[] = { 0.0, 0.0, -1.0, bbox.maxz };
    glClipPlane(GL_CLIP_PLANE0, plane0);
    glEnable(GL_CLIP_PLANE0);

    double plane1[] = { 0.0, 0.0, 1.0, -bbox.minz };
    glClipPlane(GL_CLIP_PLANE1, plane1);
    glEnable(GL_CLIP_PLANE1);

    double plane2[] = { 0.0, -1.0, 0.0, bbox.maxy };
    glClipPlane(GL_CLIP_PLANE2, plane2);
    glEnable(GL_CLIP_PLANE2);

    double plane3[] = { 0.0, 1.0, 0.0, -bbox.miny };
    glClipPlane(GL_CLIP_PLANE3, plane3);
    glEnable(GL_CLIP_PLANE3);

    double plane4[] = { -1.0, 0.0, 0.0, bbox.maxx };
    glClipPlane(GL_CLIP_PLANE4, plane4);
    glEnable(GL_CLIP_PLANE4);

    double plane5[] = { 1.0, 0.0, 0.0, -bbox.minx };
    glClipPlane(GL_CLIP_PLANE5, plane5);
    glEnable(GL_CLIP_PLANE5);
  }

  void Viewer::disableClipPlanes()
  {
    glDisable(GL_CLIP_PLANE0);
    glDisable(GL_CLIP_PLANE1);
    glDisable(GL_CLIP_PLANE2);
    glDisable(GL_CLIP_PLANE3);
    glDisable(GL_CLIP_PLANE4);
    glDisable(GL_CLIP_PLANE5);
  }

  void Viewer::normalizeScene()
  {
    switch(volumeRenderingType())
      {
      case RGBA:
	setSceneCenter(qglviewer::Vec((_rgbaVolumes[0].XMax()-_rgbaVolumes[0].XMin())/2.0+_rgbaVolumes[0].XMin(),
				      (_rgbaVolumes[0].YMax()-_rgbaVolumes[0].YMin())/2.0+_rgbaVolumes[0].YMin(),
				      (_rgbaVolumes[0].ZMax()-_rgbaVolumes[0].ZMin())/2.0+_rgbaVolumes[0].ZMin()));
	setSceneRadius(std::max(_rgbaVolumes[0].XMax()-_rgbaVolumes[0].XMin(),
			   std::max(_rgbaVolumes[0].YMax()-_rgbaVolumes[0].YMin(),
			       _rgbaVolumes[0].ZMax()-_rgbaVolumes[0].ZMin())));
	break;
      case ColorMapped:
	setSceneCenter(qglviewer::Vec((_cmVolume.XMax()-_cmVolume.XMin())/2.0+_cmVolume.XMin(),
				      (_cmVolume.YMax()-_cmVolume.YMin())/2.0+_cmVolume.YMin(),
				      (_cmVolume.ZMax()-_cmVolume.ZMin())/2.0+_cmVolume.ZMin()));
	setSceneRadius(std::max(_cmVolume.XMax()-_cmVolume.XMin(),
			   std::max(_cmVolume.YMax()-_cmVolume.YMin(),
			       _cmVolume.ZMax()-_cmVolume.ZMin())));
	break;
      }

    if(hasBeenInitialized()) showEntireScene();
  }

  void Viewer::drawHandle(const qglviewer::Vec& from, const qglviewer::Vec& to, 
			  float radius, int nbSubdivisions, 
			  int headName, int shaftName)
  {
    GLUquadric* quadric = gluNewQuadric();

    glPushMatrix();
    glTranslatef(from[0],from[1],from[2]);
    glMultMatrixd(qglviewer::Quaternion(qglviewer::Vec(0,0,1), to-from).matrix());

    double length = (to-from).norm();
    if (radius < 0.0) radius = 0.01 * length;
    const float head = 2.5*(radius / length) + 0.1;
    const float coneRadiusCoef = 4.0 - 5.0 * head;
    
    glPushName(shaftName);
    gluCylinder(quadric, radius, radius, length * (1.0 - head/coneRadiusCoef), nbSubdivisions, 1);
    glPopName();
    glTranslatef(0.0, 0.0, length * (1.0 - head));
    glPushName(headName);
    gluCylinder(quadric, coneRadiusCoef * radius, 0.0, head * length, nbSubdivisions, 1);
    glPopName();
    glTranslatef(0.0, 0.0, -length * (1.0 - head));
    
    glPopMatrix();

    gluDeleteQuadric(quadric);
  }

  void Viewer::drawHandle(const qglviewer::Vec& from, const qglviewer::Vec& to, 
			  float r, float g, float b,
			  int headName, int shaftName,
			  float pointSize, float lineWidth)
  {
    float bgcolor[4];

    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_LIGHTING);

    //draw shaft
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(lineWidth);
    glColor3f(r,g,b);
    glPushName(shaftName);
    glBegin(GL_LINES);
    glVertex3f(from[0],from[1],from[2]);
    glVertex3f(to[0],to[1],to[2]);
    glEnd();
    glPopName();
    glDisable(GL_LINE_SMOOTH);

    //draw head
    glPointSize(pointSize);
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgcolor);
    glColor3f(1.0-bgcolor[0],1.0-bgcolor[1],1.0-bgcolor[2]);
    glPushName(headName);
    glBegin(GL_POINTS);
    glVertex3f(to[0],to[1],to[2]);
    glEnd();
    glPopName();

    glPopAttrib();
  }

  void Viewer::mousePressEvent(QMouseEvent *e)
  {
    select(e->pos());
    if(_selectedObj != -1)
      _selectedPoint = e->pos();
    else
      QGLViewer::mousePressEvent(e);
  }

  void Viewer::mouseMoveEvent(QMouseEvent *e)
  {
    using namespace qglviewer;

    if(_selectedObj != -1)
      {
	double minx, miny, minz;
	double maxx, maxy, maxz;
	
	switch(volumeRenderingType())
	  {
	  case RGBA:
	    minx = _rgbaSubVolumeSelector.minx;
	    miny = _rgbaSubVolumeSelector.miny;
	    minz = _rgbaSubVolumeSelector.minz;
	    maxx = _rgbaSubVolumeSelector.maxx;
	    maxy = _rgbaSubVolumeSelector.maxy;
	    maxz = _rgbaSubVolumeSelector.maxz;
	    break;
	  default:;
	  case ColorMapped:
	    minx = _cmSubVolumeSelector.minx;
	    miny = _cmSubVolumeSelector.miny;
	    minz = _cmSubVolumeSelector.minz;
	    maxx = _cmSubVolumeSelector.maxx;
	    maxy = _cmSubVolumeSelector.maxy;
	    maxz = _cmSubVolumeSelector.maxz;
	    break;
	  }

	Vec center((maxx-minx)/2+minx,
		   (maxy-miny)/2+miny,
		   (maxz-minz)/2+minz);
	
	Vec screenCenter = camera()->projectedCoordinatesOf(center);
	Vec transPoint(e->pos().x(),e->pos().y(),screenCenter.z);
	Vec selectPoint(_selectedPoint.x(),_selectedPoint.y(),screenCenter.z);

	switch(_selectedObj)
	  {
	  case MaxXHead:
	  case MaxXShaft:
	    {
	      Vec maxXrayDir = Vec(maxx,center.y,center.z) - center;

	      Vec transPointWorld = camera()->unprojectedCoordinatesOf(transPoint);
	      Vec selectPointWorld = camera()->unprojectedCoordinatesOf(selectPoint);

	      Vec transPointWorldDir = transPointWorld - center;
	      Vec selectPointWorldDir = selectPointWorld - center;

	      transPointWorldDir.projectOnAxis(maxXrayDir);
	      selectPointWorldDir.projectOnAxis(maxXrayDir);
	      
	      transPointWorldDir -= selectPointWorldDir;

	      maxx += transPointWorldDir.x;

	      if(_selectedObj == MaxXShaft)
		minx += transPointWorldDir.x;
	      else
		minx -= transPointWorldDir.x;
	    }
	    break;
	  case MinXHead:
	  case MinXShaft:
	    {
	      Vec minXrayDir = Vec(minx,center.y,center.z) - center;

	      Vec transPointWorld = camera()->unprojectedCoordinatesOf(transPoint);
	      Vec selectPointWorld = camera()->unprojectedCoordinatesOf(selectPoint);

	      Vec transPointWorldDir = transPointWorld - center;
	      Vec selectPointWorldDir = selectPointWorld - center;

	      transPointWorldDir.projectOnAxis(minXrayDir);
	      selectPointWorldDir.projectOnAxis(minXrayDir);
	      
	      transPointWorldDir -= selectPointWorldDir;

	      minx += transPointWorldDir.x;

	      if(_selectedObj == MinXShaft)
		maxx += transPointWorldDir.x;
	      else
		maxx -= transPointWorldDir.x;
	    }
	    break;
	  case MaxYHead:
	  case MaxYShaft:
	    {
	      Vec maxYrayDir = Vec(center.x,maxy,center.z) - center;

	      Vec transPointWorld = camera()->unprojectedCoordinatesOf(transPoint);
	      Vec selectPointWorld = camera()->unprojectedCoordinatesOf(selectPoint);

	      Vec transPointWorldDir = transPointWorld - center;
	      Vec selectPointWorldDir = selectPointWorld - center;

	      transPointWorldDir.projectOnAxis(maxYrayDir);
	      selectPointWorldDir.projectOnAxis(maxYrayDir);
	      
	      transPointWorldDir -= selectPointWorldDir;

	      maxy += transPointWorldDir.y;

	      if(_selectedObj == MaxYShaft)
		miny += transPointWorldDir.y;
	      else
		miny -= transPointWorldDir.y;
	    }
	    break;
	  case MinYHead:
	  case MinYShaft:
	    {
	      Vec minYrayDir = Vec(center.x,miny,center.z) - center;

	      Vec transPointWorld = camera()->unprojectedCoordinatesOf(transPoint);
	      Vec selectPointWorld = camera()->unprojectedCoordinatesOf(selectPoint);

	      Vec transPointWorldDir = transPointWorld - center;
	      Vec selectPointWorldDir = selectPointWorld - center;

	      transPointWorldDir.projectOnAxis(minYrayDir);
	      selectPointWorldDir.projectOnAxis(minYrayDir);
	      
	      transPointWorldDir -= selectPointWorldDir;

	      miny += transPointWorldDir.y;

	      if(_selectedObj == MinYShaft)
		maxy += transPointWorldDir.y;
	      else
		maxy -= transPointWorldDir.y;
	    }
	    break;
	  case MaxZHead:
	  case MaxZShaft:
	    {
	      Vec maxZrayDir = Vec(center.x,center.y,maxz) - center;

	      Vec transPointWorld = camera()->unprojectedCoordinatesOf(transPoint);
	      Vec selectPointWorld = camera()->unprojectedCoordinatesOf(selectPoint);

	      Vec transPointWorldDir = transPointWorld - center;
	      Vec selectPointWorldDir = selectPointWorld - center;

	      transPointWorldDir.projectOnAxis(maxZrayDir);
	      selectPointWorldDir.projectOnAxis(maxZrayDir);
	      
	      transPointWorldDir -= selectPointWorldDir;

	      maxz += transPointWorldDir.z;

	      if(_selectedObj == MaxZShaft)
		minz += transPointWorldDir.z;
	      else
		minz -= transPointWorldDir.z;
	    }
	    break;
	  case MinZHead:
	  case MinZShaft:
	    {
	      Vec minZrayDir = Vec(center.x,center.y,minz) - center;

	      Vec transPointWorld = camera()->unprojectedCoordinatesOf(transPoint);
	      Vec selectPointWorld = camera()->unprojectedCoordinatesOf(selectPoint);

	      Vec transPointWorldDir = transPointWorld - center;
	      Vec selectPointWorldDir = selectPointWorld - center;

	      transPointWorldDir.projectOnAxis(minZrayDir);
	      selectPointWorldDir.projectOnAxis(minZrayDir);
	      
	      transPointWorldDir -= selectPointWorldDir;

	      minz += transPointWorldDir.z;

	      if(_selectedObj == MinZShaft)
		maxz += transPointWorldDir.z;
	      else
		maxz -= transPointWorldDir.z;
	    }
	    break;
	  }

	//if we ran into trouble with our above math, abort this bounding box change
#ifndef WIN32
	if(std::isnan(minx) || std::isnan(miny) || std::isnan(minz) ||
	   std::isnan(maxx) || std::isnan(maxy) || std::isnan(maxz))
#else
	if(_isnan(minx) || _isnan(miny) || _isnan(minz) ||
	   _isnan(maxx) || _isnan(maxy) || _isnan(maxz))
#endif
	  return;

	switch(volumeRenderingType())
	  {
	  case RGBA:
	    _rgbaSubVolumeSelector.minx = std::min(std::max(minx,_rgbaVolumes[0].XMin()),_rgbaVolumes[0].XMax());
	    _rgbaSubVolumeSelector.miny = std::min(std::max(miny,_rgbaVolumes[0].YMin()),_rgbaVolumes[0].YMax());
	    _rgbaSubVolumeSelector.minz = std::min(std::max(minz,_rgbaVolumes[0].ZMin()),_rgbaVolumes[0].ZMax());
	    _rgbaSubVolumeSelector.maxx = std::max(std::min(maxx,_rgbaVolumes[0].XMax()),_rgbaVolumes[0].XMin());
	    _rgbaSubVolumeSelector.maxy = std::max(std::min(maxy,_rgbaVolumes[0].YMax()),_rgbaVolumes[0].YMin());
	    _rgbaSubVolumeSelector.maxz = std::max(std::min(maxz,_rgbaVolumes[0].ZMax()),_rgbaVolumes[0].ZMin());
	    break;
	  default:;
	  case ColorMapped:
	    _cmSubVolumeSelector.minx = std::min(std::max(minx,_cmVolume.XMin()),_cmVolume.XMax());
	    _cmSubVolumeSelector.miny = std::min(std::max(miny,_cmVolume.YMin()),_cmVolume.YMax());
	    _cmSubVolumeSelector.minz = std::min(std::max(minz,_cmVolume.ZMin()),_cmVolume.ZMax());
	    _cmSubVolumeSelector.maxx = std::max(std::min(maxx,_cmVolume.XMax()),_cmVolume.XMin());
	    _cmSubVolumeSelector.maxy = std::max(std::min(maxy,_cmVolume.YMax()),_cmVolume.YMin());
	    _cmSubVolumeSelector.maxz = std::max(std::min(maxz,_cmVolume.ZMax()),_cmVolume.ZMin());

// 	    qDebug("(%f,%f,%f) (%f,%f,%f)",
// 	       _cmSubVolumeSelector.minx,_cmSubVolumeSelector.miny,_cmSubVolumeSelector.minz,
// 	       _cmSubVolumeSelector.maxx,_cmSubVolumeSelector.maxy,_cmSubVolumeSelector.maxz);
	    break;
	  }

	_selectedPoint = e->pos();

	if(hasBeenInitialized()) updateGL();
      }
    else
      QGLViewer::mouseMoveEvent(e);
  }

  void Viewer::mouseReleaseEvent(QMouseEvent *e)
  {
    if(_selectedObj != -1)
      {
	_selectedObj = -1;
	
	//lets normalize the bounding boxes in case they're inverted (i.e. min > max)
	_rgbaSubVolumeSelector.normalize();
	_cmSubVolumeSelector.normalize();
	
//     qDebug("(%f,%f,%f) (%f,%f,%f)",
// 	   _cmSubVolumeSelector.minx,_cmSubVolumeSelector.miny,_cmSubVolumeSelector.minz,
// 	   _cmSubVolumeSelector.maxx,_cmSubVolumeSelector.maxy,_cmSubVolumeSelector.maxz);
	
	VolMagick::BoundingBox emitbox;
	switch(volumeRenderingType())
	  {
	  case RGBA:
	    emitbox = _rgbaSubVolumeSelector;
	    break;
	  case ColorMapped:
	    emitbox = _cmSubVolumeSelector;
	    break;
	  }

	emit subVolumeSelectorChanged(emitbox);
      }

    QGLViewer::mouseReleaseEvent(e);
  }
}
