/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Skeletonization.

  Skeletonization is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  Skeletonization is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef __SKELETONIZATION__OP_H__
#define __SKELETONIZATION__OP_H__

#include <Skeletonization/datastruct.h>
#include <Skeletonization/util.h>
#include <Skeletonization/graph.h>
#include <Skeletonization/hfn_util.h>
#include <Skeletonization/skel.h>

namespace Skeletonization
{
  void draw_ray(const Ray& ray, 
		const double& r, 
		const double& g, 
		const double& b, 
		const double& a,
		ofstream& fout);

  void draw_segment(const Segment& segment,
		    const double& r, 
		    const double& g, 
		    const double& b, 
		    const double& a,
		    ofstream& fout);

  void draw_poly(const vector<Point>& poly,
		 const double& r, 
		 const double& g, 
		 const double& b, 
		 const double& a,
		 ofstream& fout);

  void draw_VF(const Triangulation& triang,
	       const Edge& dual_e, 
	       const double& r, 
	       const double& g, 
	       const double& b, 
	       const double& a,
	       ofstream& fout);

  void draw_tetra(const Cell_handle& cell,
		  const double& r, 
		  const double& g, 
		  const double& b, 
		  const double& a,
		  ofstream& fout);

  void write_wt( const Triangulation &triang,
		 const char* file_prefix);

  void write_iobdy( const Triangulation &triang,
		    const char* file_prefix);

  void write_axis(const Triangulation& triang,
		  const char* file_prefix);

  void write_u2(const Triangulation& triang,
		const Graph &graph,
		const char* file_prefix);

  void write_u1( const Triangulation& triang,
		 const char* file_prefix);

  void write_skel( const Skel& skel,
		   const char* file_prefix);

  void write_pl_cyl(const Triangulation& triang, 
		    const char* file_prefix);

  void write_pl_cyl_surf(const Triangulation& triang, 
			 const char* file_prefix);

  void write_helices(const Triangulation& triang, 
		     map<int, cell_cluster> &cluster_set, 
		     const vector<int>& sorted_cluster_index_vector, 
		     const int& helix_cnt, const char* file_prefix);
}
#endif // OP_H

