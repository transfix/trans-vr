/******************************************************************************

        Authors: Jose Rivera <transfix@ices.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

				Copyright   

This code is developed within the Computational Visualization Center at The 
University of Texas at Austin.

This code has been made available to you under the auspices of a Lesser General 
Public License (LGPL) (http://www.ices.utexas.edu/cvc/software/license.html) 
and terms that you have agreed to.

Upon accepting the LGPL, we request you agree to acknowledge the use of use of 
the code that results in any published work, including scientific papers, 
films, and videotapes by citing the following references:

C. Bajaj, Z. Yu, M. Auer
Volumetric Feature Extraction and Visualization of Tomographic Molecular Imaging
Journal of Structural Biology, Volume 144, Issues 1-2, October 2003, Pages 
132-143.

If you desire to use this code for a profit venture, or if you do not wish to 
accept LGPL, but desire usage of this code, please contact Chandrajit Bajaj 
(bajaj@ices.utexas.edu) at the Computational Visualization Center at The 
University of Texas at Austin for a different license.
******************************************************************************/

/* $Id: SkeletonRenderable.cpp 1528 2010-03-12 22:28:08Z transfix $ */

#include <stdlib.h>
#include <string.h>
#include <glew/glew.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <set>
#include <algorithm>
#include <utility>
#include <boost/array.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/utility.hpp>
#include <VolumeRover/SkeletonRenderable.h>
	
//#define RENDER_POLYS_DIRECTLY
		
void SkeletonRenderable::skel(const Skeletonization::Simple_skel& s)
{
  _skel = s;

  //tesselate the skeleton polygons into triangles because the geometry class
  //cannot handle arbitrary polygons...
  boost::tuple<
    std::vector<Skeletonization::Simple_vertex>,
    std::vector<unsigned int> > result = PolyTess::getTris(s.get<1>());

  //create a GeometryRenderable object that we can use to render the skeleton polygons
  _skel_polys.reset(new GeometryRenderable(new Geometry()));
  _skel_polys->getGeometry()->AllocateTris(result.get<0>().size(),
					   result.get<1>().size()/3);
  _skel_polys->getGeometry()->AllocateTriVertColors();

  for(std::vector<Skeletonization::Simple_vertex>::const_iterator i = result.get<0>().begin();
      i != result.get<0>().end();
      i++)
    {
      _skel_polys->getGeometry()->m_TriVerts[3*(i-result.get<0>().begin())+0] = i->get<0>().x();
      _skel_polys->getGeometry()->m_TriVerts[3*(i-result.get<0>().begin())+1] = i->get<0>().y();
      _skel_polys->getGeometry()->m_TriVerts[3*(i-result.get<0>().begin())+2] = i->get<0>().z();

      _skel_polys->getGeometry()->m_TriVertColors[3*(i-result.get<0>().begin())+0] = i->get<1>().get<0>();
      _skel_polys->getGeometry()->m_TriVertColors[3*(i-result.get<0>().begin())+1] = i->get<1>().get<1>();
      _skel_polys->getGeometry()->m_TriVertColors[3*(i-result.get<0>().begin())+2] = i->get<1>().get<2>();
    }
  
  for(std::vector<unsigned int>::const_iterator i = result.get<1>().begin();
      i != result.get<1>().end();
      i++)
    _skel_polys->getGeometry()->m_Tris[i-result.get<1>().begin()] = *i;
  
  //Create a GeometryRenderable object that we can use to render the skeleton lines.
  //Note, to do this we must convert the line strip into a set of independent lines.
  //And to do that we need to create an indexed list of vertices for the lines!
  typedef std::set<Skeletonization::Simple_vertex> Vertices_set;
  typedef std::vector<Skeletonization::Simple_vertex> Vertices_vec;
  typedef std::vector<Vertices_set::iterator> Line;
  typedef std::vector<unsigned int> Index_vec;
  Vertices_set vertices_set;
  std::vector<Line> handle_lines;
  for(Skeletonization::Line_strip_set::const_iterator i = skel().get<0>().begin();
      i != skel().get<0>().end();
      i++)
    {
      Line line;
      for(Skeletonization::Simple_line_strip::const_iterator j = i->begin();
	  j != i->end();
	  j++)
	{
	  std::pair<Vertices_set::iterator,bool> result = vertices_set.insert(*j);
	  line.push_back(result.first);
	}
      handle_lines.push_back(line);
    }

  //we need a random access iterator for the index calculation below
  Vertices_vec vertices_vec(vertices_set.begin(), vertices_set.end());
  std::vector<Index_vec> indices_vec;
  for(std::vector<Line>::const_iterator i = handle_lines.begin();
      i != handle_lines.end();
      i++)
    {
      Index_vec index_vec;
      for(Line::const_iterator j = i->begin();
	  j != i->end();
	  j++)
	index_vec.push_back(std::lower_bound(vertices_vec.begin(),vertices_vec.end(),**j) - vertices_vec.begin());
      indices_vec.push_back(index_vec);
    }

  //finally split up the line strips into independent lines
  Index_vec independent_lines;
  for(std::vector<Index_vec>::const_iterator i = indices_vec.begin();
      i != indices_vec.end();
      i++)
    {
      for(Index_vec::const_iterator j = i->begin();
	  j != i->end() && boost::next(j) != i->end();
	  j++)
	{
	  independent_lines.push_back(*j);
	  independent_lines.push_back(*boost::next(j));
	}
    }

  //now we are ready to fill out the _skel_lines geometry
  _skel_lines.reset(new GeometryRenderable(new Geometry()));
  _skel_lines->getGeometry()->AllocateLines(vertices_vec.size(),
					    independent_lines.size()/2);
  _skel_lines->getGeometry()->AllocateLineColors();
  for(Vertices_vec::const_iterator i = vertices_vec.begin();
      i != vertices_vec.end();
      i++)
    {
      _skel_lines->getGeometry()->m_LineVerts[3*(i-vertices_vec.begin())+0] = i->get<0>().x();
      _skel_lines->getGeometry()->m_LineVerts[3*(i-vertices_vec.begin())+1] = i->get<0>().y();
      _skel_lines->getGeometry()->m_LineVerts[3*(i-vertices_vec.begin())+2] = i->get<0>().z();

      _skel_lines->getGeometry()->m_LineColors[3*(i-vertices_vec.begin())+0] = i->get<1>().get<0>();
      _skel_lines->getGeometry()->m_LineColors[3*(i-vertices_vec.begin())+1] = i->get<1>().get<1>();
      _skel_lines->getGeometry()->m_LineColors[3*(i-vertices_vec.begin())+2] = i->get<1>().get<2>();
    }
  
  memcpy(_skel_lines->getGeometry()->m_Lines.get(),
	 &(independent_lines[0]),
	 independent_lines.size()*sizeof(unsigned int));
}

bool SkeletonRenderable::render()
{
  double scale_x,scale_y,scale_z;
  double trans_x,trans_y,trans_z;

  VolMagick::BoundingBox renderbox = _subVolumeBoundingBox;
  
  double dx = renderbox.maxx - renderbox.minx;
  double dy = renderbox.maxy - renderbox.miny;
  double dz = renderbox.maxz - renderbox.minz;

  boost::array<double,3> dV = { dx, dy, dz };

  double aspectx = dx/(*std::max_element(dV.begin(),dV.end()));
  double aspecty = dy/(*std::max_element(dV.begin(),dV.end()));
  double aspectz = dz/(*std::max_element(dV.begin(),dV.end()));

  renderbox.minx = -aspectx/2.0; renderbox.maxx = aspectx/2.0;
  renderbox.miny = -aspecty/2.0; renderbox.maxy = aspecty/2.0;
  renderbox.minz = -aspectz/2.0; renderbox.maxz = aspectz/2.0;

  scale_x = aspectx/(_subVolumeBoundingBox.XMax()-_subVolumeBoundingBox.XMin());
  scale_y = aspecty/(_subVolumeBoundingBox.YMax()-_subVolumeBoundingBox.YMin());
  scale_z = aspectz/(_subVolumeBoundingBox.ZMax()-_subVolumeBoundingBox.ZMin());
  trans_x = (-_subVolumeBoundingBox.XMin())*scale_x - aspectx/2.0;
  trans_y = (-_subVolumeBoundingBox.YMin())*scale_y - aspecty/2.0;
  trans_z = (-_subVolumeBoundingBox.ZMin())*scale_z - aspectz/2.0;

  glPushAttrib(GL_ALL_ATTRIB_BITS);
  glPushMatrix();

  if(_clipGeometry)
    setClipPlanes();

  //begin!!
  glTranslatef(trans_x,trans_y,trans_z);
  glScalef(scale_x,scale_y,scale_z);

  glDisable(GL_LIGHTING);
  glLineWidth(1.0);
#ifdef RENDER_POLYS_DIRECTLY
  for(Skeletonization::Line_strip_set::const_iterator i = _skel.get<0>().begin();
      i != _skel.get<0>().end();
      i++)
    {
      glBegin(GL_LINE_STRIP);
      for(Skeletonization::Simple_line_strip::const_iterator j = i->begin();
	  j != i->end();
	  j++)
	{
	  glColor4d(j->get<1>().get<0>(),
		    j->get<1>().get<1>(),
		    j->get<1>().get<2>(),
		    j->get<1>().get<3>());
	  glVertex3d(j->get<0>().x(),
		     j->get<0>().y(),
		     j->get<0>().z());
	}
      glEnd();
    }
#else
  if(_skel_lines) _skel_lines->render();
#endif
  
#ifdef RENDER_POLYS_DIRECTLY
  for(Skeletonization::Polygon_set::const_iterator i = _skel.get<1>().begin();
      i != _skel.get<1>().end();
      i++)
    {
      glBegin(GL_POLYGON);
      for(Skeletonization::Simple_polygon::const_iterator j = i->begin();
	  j != i->end();
	  j++)
	{
	  glColor4d(j->get<1>().get<0>(),
		    j->get<1>().get<1>(),
		    j->get<1>().get<2>(),
		    j->get<1>().get<3>());
	  glVertex3d(j->get<0>().x(),
		     j->get<0>().y(),
		     j->get<0>().z());
	}
      glEnd();
    }
#else
  glEnable(GL_LIGHTING);
  if(_skel_polys) _skel_polys->render();
#endif

  if(_clipGeometry)
    disableClipPlanes();

  glPopMatrix();
  glPopAttrib();

  return true;
}

void SkeletonRenderable::setClipPlanes()
{
  VolMagick::BoundingBox renderbox = _subVolumeBoundingBox;
  
  double dx = renderbox.maxx - renderbox.minx;
  double dy = renderbox.maxy - renderbox.miny;
  double dz = renderbox.maxz - renderbox.minz;

  boost::array<double,3> dV = { dx, dy, dz };

  double aspectx = dx/(*std::max_element(dV.begin(),dV.end()));
  double aspecty = dy/(*std::max_element(dV.begin(),dV.end()));
  double aspectz = dz/(*std::max_element(dV.begin(),dV.end()));

  double plane0[] = { 0.0, 0.0, -1.0, aspectz/2.0 + 0.00001 };
  glClipPlane(GL_CLIP_PLANE0, plane0);
  glEnable(GL_CLIP_PLANE0);
  
  double plane1[] = { 0.0, 0.0, 1.0, aspectz/2.0 + 0.00001 };
  glClipPlane(GL_CLIP_PLANE1, plane1);
  glEnable(GL_CLIP_PLANE1);
  
  double plane2[] = { 0.0, -1.0, 0.0, aspecty/2.0 + 0.00001 };
  glClipPlane(GL_CLIP_PLANE2, plane2);
  glEnable(GL_CLIP_PLANE2);

  double plane3[] = { 0.0, 1.0, 0.0, aspecty/2.0 + 0.00001 };
  glClipPlane(GL_CLIP_PLANE3, plane3);
  glEnable(GL_CLIP_PLANE3);
  
  double plane4[] = { -1.0, 0.0, 0.0, aspectx/2.0 + 0.00001 };
  glClipPlane(GL_CLIP_PLANE4, plane4);
  glEnable(GL_CLIP_PLANE4);
  
  double plane5[] = { 1.0, 0.0, 0.0, aspectx/2.0 + 0.00001 };
  glClipPlane(GL_CLIP_PLANE5, plane5);
  glEnable(GL_CLIP_PLANE5);
}

void SkeletonRenderable::disableClipPlanes()
{
  glDisable(GL_CLIP_PLANE0);
  glDisable(GL_CLIP_PLANE1);
  glDisable(GL_CLIP_PLANE2);
  glDisable(GL_CLIP_PLANE3);
  glDisable(GL_CLIP_PLANE4);
  glDisable(GL_CLIP_PLANE5);
}
