/*
  Copyright 2008-2011 The University of Texas at Austin

        Authors: Joe Rivera <transfix@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of VolumeRover.

  VolumeRover is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  VolumeRover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: cvcraw_geometry.h 5338 2012-04-04 17:15:25Z zqyork $ */

/*
  Joe R. - transfix@ices.utexas.edu - v0.7

  This library provides some useful facilities for CVC geometry I/O.
  It provides a geometry type called cvcraw_geometry::geomery_t to collect
  geometry information.  All information is contained in STL conforming containers,
  except the boundary information which uses boost::dynamic_bitset<> (I don't think
  dynamic_bitset<> can be used like an STL container exactly).

  Currently, the geometry_t class supplies 8 operations:

  calculate_extents() - simply calculates the smallest bounding box such that
                        all vertices fit inside that box.
  merge()             - merges *this with another geometry_t object
  tri_surface()       - extracts the boundary of *this and returns it as a
                        triangulated surface.  This is useful for code that
			only works with tri surfaces.
  calculate_surf_normals() - simply calculates a normal for each point based on an
                        approximate surface normal for each boundary element.  If points
			are unreferenced or are non-boundary, the normal is set to 
			0.0,0.0,0.0 for now.
  generate_wire_interior() - returns a representation of an interior mesh such that
                        internal faces are converted to lines.  Useful for visualizing
			such meshes.
  invert_normals()    - simply reverses all normals
  reorient()          - An attempt at fixing adjacent normal issues.  Doesn't quite work
                        yet.
  clear()             - clears all geometry

  Note: I am considering moving these operations outside of geometry_t and instead
  implementing them as template functions so they aren't dependent on the
  cvcraw_geometry::geometry_t type exactly.  However I need to spend some more time
  thinking about what an appropriate set of traits would be for generic geometry types.
                  
  The library also provides the following I/O template functions:
  read()              - reads the geometry from the filename provided and stores it
                        into the container object provided.  It figures out what kind
			of geometry is present in the file based on it's structure.
			It supports all CVC mesh types including those used by LBIE.
  write()             - writes the geometry to the filename provided.  It will write
                        the correct kind of geometry file depending on what information
			is present in the geometry container.

  If there is an error during reading or writing, an exception will be thrown.

  Example usage:

  #include <cstdlib>
  #include <iostream>
  #include <cvcraw_geometry/cvcraw_geometry.h>

  int main(int argc, char **argv)
  {
    using namespace std;
    try
     {  
       cvcraw_geometry::geometry_t geom;
       cvcraw_geometry::read(geom, "test.raw");
       cvcraw_geometry::write(geom.tri_surface(), "output.raw"); //output surface triangulation
     }
    catch(std::exception& e)
     {
       cout << "Error: " << e.what() << endl;
       return EXIT_FAILURE;
     }

    return EXIT_SUCCESS;
  }
*/

/*
  Change Log:

  10/08/2008 - v0.1 - initial version
  10/12/2008 - v0.2 - added support for indices starting at zero or one
  10/27/2008 - v0.3 - added calculate_surf_normals()
                      also added CVCRAW_GEOMETRY__CORRECT_INDEX_START to enable
		      fixing indices starting at 1 since enabling it all the time
		      breaks some meshes...
  10/31/2008 - v0.4 - forgot to copy boundary info in geometry_t copy constructor
                      and operator=().  Now returning refrence to *this in calculate_
		      functions.
  01/30/2009 - v0.5 - adding trim functions and compressing adjacent tokens to support
                      more raw files with extra space characters
  02/09/2009 - v0.6 - adding empty() call
  09/18/2009 - v0.7 - adding clear() call, using uint64_t for indicies
  04/02/2010 - v1.0 - creating it's own library and breaking up the single header
                      into multiple headers.
*/

#ifndef __CVCRAW_GEOMETRY_H__
#define __CVCRAW_GEOMETRY_H__

#include <cvcraw_geometry/utility.h>
#include <cvcraw_geometry/io.h>

#include <boost/cstdint.hpp>
#include <boost/array.hpp>
#include <boost/dynamic_bitset.hpp>

namespace cvcraw_geometry
{
  typedef boost::uint64_t uint64_t;

  class geometry_t
  {
  public:
    typedef boost::array<double,3> point_t;
    typedef boost::array<double,3> normal_t;
    typedef boost::array<double,3> vector_t;
    typedef boost::array<double,3> color_t;
    typedef boost::array<uint64_t,2>    line_t;
    typedef boost::array<uint64_t,3>    triangle_t;
    typedef boost::array<uint64_t,4>    quad_t;

    typedef std::vector<point_t>    points_t;
    typedef boost::dynamic_bitset<> boundary_t;
    typedef std::vector<normal_t>   normals_t;
    typedef std::vector<color_t>    colors_t;
    typedef std::vector<line_t>     lines_t;
    typedef std::vector<triangle_t> triangles_t;
    typedef std::vector<quad_t>     quads_t;

    geometry_t();

    geometry_t(const geometry_t& copy);
    ~geometry_t();

    geometry_t& operator=(const geometry_t& copy);

    points_t    points;
    boundary_t  boundary;
    normals_t   normals;
    colors_t    colors;
    lines_t     lines;
    triangles_t tris;
    quads_t     quads;

    point_t min_ext;
    point_t max_ext;

    point_t min_point() const { return min_ext; }
    point_t max_point() const { return max_ext; }

    bool empty() const;

    geometry_t& calculate_extents();
    geometry_t& merge(const geometry_t& geom);

    //returns a simple tri surface for the boundary
    //doesn't remove extra non boundary points
    geometry_t tri_surface() const;

    //calculates normals for boundary vertices
    //sets non boundary vertex normals to 0.0,0.0,0.0 until further notice
    geometry_t& calculate_surf_normals();

	//calculates normals with angle weights
	geometry_t& calculate_surf_normals_angle_weight();

    //This is a little hack to get a simple tetra or hex mesh rendering,
    //using the lines array to draw an internal wireframe
    geometry_t generate_wire_interior() const;

    //simply inverts all the normals
    geometry_t& invert_normals();

    //makes normals consistent... TODO: make this actually re-orient triangles
    //(i.e. CCW => CW or CW => CCW depending on which direction we need the normal
    geometry_t& reorient();

    //Clears this object
    geometry_t& clear();
  };
}

#endif
