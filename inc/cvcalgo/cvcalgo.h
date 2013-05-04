/*
  Copyright 2008 The University of Texas at Austin

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

/* $Id: cvcalgo.h 4741 2011-10-21 21:22:06Z transfix $ */

#ifndef __CVCALGO_CVCALGO_H__
#define __CVCALGO_CVCALGO_H__

#ifndef NO_LBIE
#include <LBIE/LBIE_Mesher.h>
#endif

#ifndef NO_FASTCONTOURING
#include <FastContouring/FastContouring.h>
#endif


// arand, 5-6-2011: commenting the NO_VOLUMEROVER_EXTENSION stuff
//                  this code seems to be out of date
//#ifndef NO_VOLUMEROVER_EXTENSION
//#include <VolumeRover2/VolumeRoverExtension.h>
//#include <VolumeRover2/VolumeRoverMain.h>
//#endif

#include <cvcraw_geometry/cvcgeom.h>
#include <cvcraw_geometry/Geometry.h>
#include <VolMagick/VolMagick.h>

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/array.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/cstdint.hpp>

/*
  This library is a front end to many cvc algorithms used by VolumeRover.

  Geometry data is stored in cvcraw_geometry::cvcgeom_t objects and volume data is in
  VolMagick::Volume objects.
*/

namespace cvcalgo
{

  /*
#ifndef NO_VOLUMEROVER_EXTENSION
  class CVCAlgoExtension : public VolumeRoverExtension
  {
  public:
    virtual void init();
  };


  extern boost::shared_ptr<CVCAlgoExtension> _instance;

  boost::shared_ptr<CVCAlgoExtension> instance();

  extern void init();
#endif
  */

#ifndef NO_LBIE
  cvcraw_geometry::cvcgeom_t convert(const LBIE::geoframe& geo);
#endif
#ifndef NO_FASTCONTOURING
  cvcraw_geometry::cvcgeom_t convert(const FastContouring::TriSurf& geo);
#endif  

  typedef boost::uint64_t uint64_t;

  typedef cvcraw_geometry::cvcgeom_t::point_t point_t;
  typedef cvcraw_geometry::cvcgeom_t::vector_t vector_t;
  typedef cvcraw_geometry::cvcgeom_t::color_t color_t;
  
  /*
    Conversion routines for Geometry <=> cvcgeom_t
  */
  Geometry convert(const cvcraw_geometry::cvcgeom_t& src);
  cvcraw_geometry::cvcgeom_t convert(const Geometry& src);


  enum SDFMethod { MULTI_SDF, SDFLIB };
  /*
    Compute a signed distance field using the arguments specified.
  */
  VolMagick::Volume sdf(const cvcraw_geometry::cvcgeom_t& geom,
                        /*
                          Dimension of output sdf vol.
                        */
                        const VolMagick::Dimension& dim,
                        /*
                          Bounding box of output vol. If default initialized,
                          use extents of Geometry.
                        */
                        const VolMagick::BoundingBox& bbox = VolMagick::BoundingBox(),
                        /*
                          The SDF library to use to generate the volume.
                        */
                        SDFMethod method = MULTI_SDF);


  /*
   * volren - Volume raycaster interface
   */
  class VolrenParameters
  {
  public:

    VolrenParameters() :
      _perspective(true),
      _fov(45.0)
        {
          _cameraPosition[0] = 0.0;
          _cameraPosition[1] = 0.0;
          _cameraPosition[2] = -1000.0;

          _viewUpVector[0] = 0.0;
          _viewUpVector[1] = 1.0;
          _viewUpVector[2] = 0.0;

          _viewPlaneNormal[0] = 0.0;
          _viewPlaneNormal[1] = 0.0;
          _viewPlaneNormal[2] = 1.0;

          _viewPlaneResolution[0] = 512;
          _viewPlaneResolution[1] = 512;

          _finalImagePixelResolution[0] = 512;
          _finalImagePixelResolution[1] = 512;
        }

    VolrenParameters(const VolrenParameters& copy)
      {
        _perspective = copy._perspective;
        _fov = copy._fov;
        _cameraPosition = copy._cameraPosition;
        _viewUpVector = copy._viewUpVector;
        _viewPlaneNormal = copy._viewPlaneNormal;
        for(int i = 0; i < 2; i++)
          _viewPlaneResolution[i] = copy._viewPlaneResolution[i];
        for(int i = 0; i < 2; i++)
          _finalImagePixelResolution[i] = copy._finalImagePixelResolution[i];
      }

    // camera settings
    bool perspective() const { return _perspective; }
    VolrenParameters& perspective(bool flag) { _perspective = flag; return *this; }
    float fov() const { return _fov; }
    VolrenParameters& fov(float val) { _fov = val; return *this; }
    point_t cameraPosition() const { return _cameraPosition; }
    VolrenParameters& cameraPosition(const point_t& p) { _cameraPosition = p; return *this; }
    vector_t viewUpVector() const { return _viewUpVector; }
    VolrenParameters& viewUpVector(const vector_t& v) { _viewUpVector = v; return *this; }
    vector_t viewPlaneNormal() const { return _viewPlaneNormal; }
    VolrenParameters& viewPlaneNormal(const vector_t& v) { _viewPlaneNormal = v; return *this; }
    const uint64_t* viewPlaneResolution() const { return _viewPlaneResolution; }
    template<class C>
      VolrenParameters& viewPlaneResolution(const C& v)
      {
        _viewPlaneResolution[0] = v[0];
        _viewPlaneResolution[1] = v[1];
        return *this;
      }
    const uint64_t* finalImagePixelResolution() const { return _finalImagePixelResolution; }
    template<class C>
      VolrenParameters& finalImagePixelResolution(const C& v)
      {
        _finalImagePixelResolution[0] = v[0];
        _finalImagePixelResolution[1] = v[1];
        return *this;
      }

    //material settings
    

  private:
    bool _perspective;
    float _fov;
    point_t _cameraPosition;
    vector_t _viewUpVector;
    vector_t _viewPlaneNormal;
    uint64_t _viewPlaneResolution[2];
    uint64_t _finalImagePixelResolution[2];
  };

  typedef boost::shared_array<unsigned char> Image;
  
  
}

#endif
