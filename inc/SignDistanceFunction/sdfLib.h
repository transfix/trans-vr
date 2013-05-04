/*
  Copyright 2004-2005 The University of Texas at Austin

        Authors: Lalit Karlapalem <ckl@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of SignDistanceFunction.

  SignDistanceFunction is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  SignDistanceFunction is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

/* $Id: sdfLib.h 4741 2011-10-21 21:22:06Z transfix $ */

#ifndef CCV_SDF_SDFLIB_H
#define CCV_SDF_SDFLIB_H

#include <boost/shared_ptr.hpp>
#include <VolMagick/VolMagick.h>
#include <cvcraw_geometry/Geometry.h>

namespace SDFLibrary {

//First, set the parameters of the SDF grid.
void setParameters(int size, int isNormalFlip, float* mins, float* maxs);

typedef struct RAWIV_header
{
	float minext[3];	//Co-ords of the first voxel
	float maxext[3];	//Co-ords of the last voxel
	float origin[3];	//Co-ords of the first voxel a.k.a. Origin
	float span[3];		//Span between grid points
	int dim[3];			//number of grid points

	int ngridpts;		//Total grid points
	int ncells;			//Total cells
	int size;			//Octree size

}RAWIV_header;


//Then, call the function with the input triangulated data. The SDF values are returned
float* computeSDF(int nverts, float* verts, int ntris, int* tris);

RAWIV_header* getVolumeInfo();

//Added for VolRover - Joe R. <transfix@ices.utexas.edu>
 VolMagick::Volume signedDistanceFunction(/*
					    Input geometry for sdf.
					  */
					  const boost::shared_ptr<Geometry>& geom,
					  /*
					    Dimension of output sdf vol.
					  */
					  const VolMagick::Dimension& dim,
					  /*
					    Bounding box of output vol. If default initialized,
					    use extents of Geometry.
					  */
					  const VolMagick::BoundingBox& bbox = VolMagick::BoundingBox());
};

#endif
