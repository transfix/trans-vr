/******************************************************************************

        Authors: Samrat Goswami <tarmas@gmail.com>
                 Jose Rivera <transfix@ices.utexas.edu>
                 Ajay Gopinath <ajay.gopinath@gmail.com>
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

/* $Id: SkeletonRenderable.h 1527 2010-03-12 22:10:16Z transfix $ */

#ifndef __VOLUME__SKELETONRENDERABLE_H__
#define __VOLUME__SKELETONRENDERABLE_H__

#include <VolumeWidget/Renderable.h>
#include <VolumeWidget/GeometryRenderable.h>
#include <VolMagick/VolMagick.h>
#include <boost/shared_ptr.hpp>
#include <Skeletonization/Skeletonization.h>

class SkeletonRenderable : public Renderable
{
 public:
  SkeletonRenderable(const Skeletonization::Simple_skel& s = Skeletonization::Simple_skel())
    : _skel(s), _clipGeometry(true) {}
  ~SkeletonRenderable() {}

  void skel(const Skeletonization::Simple_skel& s);
  Skeletonization::Simple_skel skel() const { return _skel; }
  Skeletonization::Simple_skel& skel() { return _skel; }

  void setSubVolume(const VolMagick::BoundingBox& subvolbox) { _subVolumeBoundingBox = subvolbox; }

  virtual bool render();

  void setClipGeometry(bool clip) { _clipGeometry = clip; }

  const boost::shared_ptr<GeometryRenderable> skel_polys() const { return _skel_polys; }
  const boost::shared_ptr<GeometryRenderable> skel_lines() const { return _skel_lines; }

  void clear()
  { 
    _skel = Skeletonization::Simple_skel();
    _skel_polys.reset();
    _skel_lines.reset();
  }

 private:
  void setClipPlanes();
  void disableClipPlanes();

  Skeletonization::Simple_skel _skel;
  VolMagick::BoundingBox _subVolumeBoundingBox; // defines the subvolume to clip this rendering with

  boost::shared_ptr<GeometryRenderable> _skel_polys; //use this object to render the skeleton polygons
  boost::shared_ptr<GeometryRenderable> _skel_lines; //use this object to render the skeleton lines

  bool _clipGeometry;
};

#endif

