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

#ifndef __CVCGEOM_H__
#define __CVCGEOM_H__

#ifndef CVCGEOM_NAMESPACE
#define CVCGEOM_NAMESPACE cvcraw_geometry
#endif

#include <boost/cstdint.hpp>
#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/dynamic_bitset.hpp>

#include <vector>

#ifndef DISABLE_CONVERSION
namespace CVCGEOM_NAMESPACE
{
  class geometry_t;
}
#endif

namespace CVCGEOM_NAMESPACE
{
  typedef boost::uint64_t uint64_t;

  // --------
  // cvcgeom_t
  // --------
  // Purpose: 
  //   Standard geometry container for cvc algorithms.
  // ---- Change History ----
  // 07/03/2010 -- Joe R. -- Initial implementation.
  class cvcgeom_t
  {
  public:
    typedef double                         scalar_t;
    typedef uint64_t                       index_t;
    typedef boost::array<scalar_t,3>       point_t;
    typedef boost::array<scalar_t,3>       vector_t;
    typedef vector_t                       normal_t;
    typedef boost::array<scalar_t,3>       color_t;
    typedef boost::array<index_t,2>        line_t;
    typedef boost::array<index_t,3>        triangle_t;
    typedef boost::array<index_t,4>        quad_t;

    typedef std::vector<point_t>           points_t;
    typedef boost::dynamic_bitset<>        boundary_t;
    typedef std::vector<vector_t>          normals_t;
    typedef std::vector<color_t>           colors_t;
    typedef std::vector<line_t>            lines_t;
    typedef std::vector<triangle_t>        triangles_t;
    typedef std::vector<quad_t>            quads_t;

    typedef boost::shared_ptr<points_t>    points_ptr_t;
    typedef boost::shared_ptr<boundary_t>  boundary_ptr_t;
    typedef boost::shared_ptr<normals_t>   normals_ptr_t;
    typedef boost::shared_ptr<colors_t>    colors_ptr_t;
    typedef boost::shared_ptr<lines_t>     lines_ptr_t;
    typedef boost::shared_ptr<triangles_t> triangles_ptr_t;
    typedef boost::shared_ptr<quads_t>     quads_ptr_t;

    enum ARRAY_TYPE
      {
        POINTS,
        BOUNDARY,
        NORMALS,
        COLORS,
        LINES,
        TRIANGLES,
        QUADS
      };

    cvcgeom_t();
    cvcgeom_t(const cvcgeom_t& geom);
    cvcgeom_t(const std::string & filename);
    ~cvcgeom_t();

    void read_raw(const std::string & filename);
    void read_off(const std::string & filename);


    void copy(const cvcgeom_t& geom);
    cvcgeom_t& operator=(const cvcgeom_t& geom);

#ifndef DISABLE_CONVERSION
    cvcgeom_t(const geometry_t& geom);
    void copy(const geometry_t& geom);
    cvcgeom_t& operator=(const geometry_t& geom);
    operator geometry_t() const;
#endif

    points_t&    points() { pre_write(POINTS); return *_points; }
    boundary_t&  boundary() { pre_write(BOUNDARY); return *_boundary; }
    normals_t&   normals() { pre_write(NORMALS); return *_normals; }
    colors_t&    colors() { pre_write(COLORS); return *_colors; }
    lines_t&     lines() { pre_write(LINES); return *_lines; }
    triangles_t& triangles() { pre_write(TRIANGLES); return *_triangles; }
    quads_t&     quads() { pre_write(QUADS); return *_quads; }

    const points_t&    const_points() const { return *_points; }
    const boundary_t&  const_boundary() const { return *_boundary; }
    const normals_t&   const_normals() const { return *_normals; }
    const colors_t&    const_colors() const { return *_colors; }
    const lines_t&     const_lines() const { return *_lines; }
    const triangles_t& const_triangles() const { return *_triangles; }
    const quads_t&     const_quads() const { return *_quads; }

    const points_t&    points() const { return const_points(); }
    const boundary_t&  boundary() const { return const_boundary(); }
    const normals_t&   normals() const { return const_normals(); }
    const colors_t&    colors() const { return const_colors(); }
    const lines_t&     lines() const { return const_lines(); }
    const triangles_t& triangles() const { return const_triangles(); }
    const quads_t&     quads() const { return const_quads(); }
    
    point_t min_point() const;
    point_t max_point() const;

    uint64_t num_points() const;
    uint64_t num_lines() const;
    uint64_t num_triangles() const;
    uint64_t num_quads() const;

    bool empty() const;
    
    cvcgeom_t& merge(const cvcgeom_t& geom);
    
    //returns a simple tri surface for the boundary
    //doesn't remove extra non boundary points
    cvcgeom_t tri_surface() const;

    //calculates normals for boundary vertices
    //sets non boundary vertex normals to 0.0,0.0,0.0 until further notice
    cvcgeom_t& calculate_surf_normals();

    //This is a little hack to get a simple tetra or hex mesh rendering,
    //using the lines array to draw an internal wireframe
    cvcgeom_t generate_wire_interior() const;

    //simply inverts all the normals
    cvcgeom_t& invert_normals();

    //makes normals consistent... TODO: make this actually re-orient triangles
    //(i.e. CCW => CW or CW => CCW depending on which direction we need the normal
    cvcgeom_t& reorient();

    //Clears this object
    cvcgeom_t& clear();

    //Projects boundary vertices of this geometry to the input geometry
    cvcgeom_t& project(const cvcgeom_t &input);

  protected:
    void init_ptrs();
    void calc_extents() const;
    virtual void pre_write(ARRAY_TYPE at);

    points_ptr_t    _points;
    boundary_ptr_t  _boundary;
    normals_ptr_t   _normals;
    colors_ptr_t    _colors;
    lines_ptr_t     _lines;
    triangles_ptr_t _triangles;
    quads_ptr_t     _quads;

    //calculated on demand even for const, so the following must be mutable.
    mutable bool _extents_set; //if true, the min/max extents are valid
    mutable point_t _min;
    mutable point_t _max;
  };

  // 06/01/2012 - returns the classic bunny mesh
  cvcgeom_t bunny();

  typedef cvcgeom_t::scalar_t    scalar_t;
  typedef cvcgeom_t::index_t     index_t;
  typedef cvcgeom_t::vector_t    vector_t;
  typedef cvcgeom_t::point_t     point_t;
  typedef cvcgeom_t::color_t     color_t;
  typedef cvcgeom_t::points_t    points_t;
  typedef cvcgeom_t::boundary_t  boundary_t;
  typedef cvcgeom_t::normals_t   normals_t;
  typedef cvcgeom_t::colors_t    colors_t;
  typedef cvcgeom_t::lines_t     lines_t;
  typedef cvcgeom_t::triangle_t triangle_t;
  typedef cvcgeom_t::triangles_t triangles_t;
  typedef cvcgeom_t::quads_t     quads_t;
}

#endif
