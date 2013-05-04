/*
  Copyright 2000-2003 The University of Texas at Austin

	Authors: Xiaoyu Zhang 2000-2002 <xiaoyu@ices.utexas.edu>
					 John Wiggins 2003-2004 <prok@ices.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of iotree.

  iotree is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  iotree is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with iotree; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
//  ______________________________________________________________________
//
//    FILE
//      contour3d.h - class for a 3d isocontour surface
//
//      Copyright (c) 1998 Emilio Camahort, Dan Schikore
//
//    DESCRIPTION
//      contour3d is a class for representing a contour surface (or any
//      3d triangular mesh).
//  ______________________________________________________________________

//
// Changed to ContourGeom by John Wiggins 2004
// 	- Removed old coloring information and replaced with a
// 	per-vertex color buffer (vcol).
// 	- Got rid of a lot of unused code. This class is only being used as
// 	a way to get triangular surfaces out of the c2c library. So now it has
// 	just enough code to accomplish that and nothing else.
//

// $Id: ContourGeom.h 1498 2010-03-10 22:50:29Z transfix $

#ifndef _CONTOUR_GEOM_H
#define _CONTOUR_GEOM_H

#include <set>
#include <c2c_codec/trimesh.h>

#ifdef WIN32
typedef unsigned int    u_int;
//#include <unitypes.h>
#else
#include <sys/types.h>
#endif

using std::set;

class ContourGeom {

public:

    // constructor
    ContourGeom();

	  // copy constructor
	  ContourGeom(const ContourGeom& con3d);

    // destructor
    ~ContourGeom();

    // add a vertex with the given position and normal
    int AddVert(float p[3], float n[3], float c[3])
                { return(AddVert(p[0], p[1], p[2],
																 n[0], n[1], n[2],
																 c[0], c[1], c[2])); }
    int AddVert(float,float,float, float,float,float, float,float,float);

    int AddVertUnique(float p[3], float n[3], float c[3])
                { return(AddVertUnique(p[0], p[1], p[2],
																			 n[0], n[1], n[2],
																			 c[0], c[1], c[2])); }

    int AddVertUnique(float,float,float, float,float,float, float,float,float);
  	int AddVertUnique(float p[3], float n[3], float c[3], EdgeIndex ei);


    // add a triangle indexed by the given 3 vertices
    int AddTri(u_int v[3])   { return(AddTri(v[0], v[1], v[2])); }
    int AddTri(u_int, u_int, u_int);

    // get the number of vertices or triangles
    int getSize(void)        { return(ntri); }
    int getNVert(void)       { return(nvert); }
    int getNTri(void)        { return(ntri);  }
    int getTsize(void)       { return(tsize); }

	  // merge the other surface into the surface
	  void merge(ContourGeom *);

protected:

    // the size of the vertex and triangle arrays
    int	vsize, tsize;

    // the number of vertices and triangles
    int	nvert, ntri;

    set<VPosition, LtPos> *pvset;
		set<EdgeIndex, LtEdge> *edgeset;

public: // made public by Emilio

    // arrays of vertices, vertex normals, and triangles
    float	(*vert)[3];			// isosurface vertex array
    float	(*vnorm)[3];		// array of vertex normals
    float	(*vcol)[3];			// color values at vertices

    unsigned int (*tri)[3];			// triangle mesh array
};

#endif

