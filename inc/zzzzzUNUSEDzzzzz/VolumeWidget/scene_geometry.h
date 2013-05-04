#ifndef __SCENE_GEOMETRY_H__
#define __SCENE_GEOMETRY_H__

#include <cvcraw_geometry/cvcraw_geometry.h>
#include <boost/shared_ptr.hpp>

//This was all ripped and backported from VolRover 2.x
struct scene_geometry_t
{
  typedef cvcraw_geometry::geometry_t geometry_t;

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

#endif
