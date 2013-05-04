/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of SecondaryStructures.

  SecondaryStructures is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  SecondaryStructures is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef __OP_H__
#define __OP_H__

#include <SecondaryStructures/hfn_util.h>
#include <SecondaryStructures/skel.h>
#include <cvcraw_geometry/cvcgeom.h>
#include <SecondaryStructures/datastruct_ss.h>

void draw_Ray(const SecondaryStructures::Ray_3 & myRay_3,
			  const double& r,
			  const double& g,
			  const double& b,
			  const double& a,
			  ofstream& fout);

void draw_segment(const SecondaryStructures::Segment& segment,
				  const double& r,
				  const double& g,
				  const double& b,
				  const double& a,
				  ofstream& fout);

void draw_poly(const vector<SecondaryStructures::Point>& poly,
			   const double& r,
			   const double& g,
			   const double& b,
			   const double& a,
			   ofstream& fout);

void draw_VF(const SecondaryStructures::Triangulation& triang,
			 const SecondaryStructures::Edge& dual_e,
			 const double& r,
			 const double& g,
			 const double& b,
			 const double& a,
			 ofstream& fout);

void draw_tetra(const SecondaryStructures::Cell_handle& cell,
				const double& r,
				const double& g,
				const double& b,
				const double& a,
				ofstream& fout);

void write_wt(const SecondaryStructures::Triangulation& triang,
			  const char* file_prefix);
void write_helix_wrl(cvcraw_geometry::cvcgeom_t* geom, char* filename);

void write_sheet_wrl(cvcraw_geometry::cvcgeom_t* geom, char* filename);
void write_iobdy(const SecondaryStructures::Triangulation& triang,
				 const char* file_prefix);

void write_axis(const SecondaryStructures::Triangulation& triang,
				const int& biggest_medax_comp_id,
				const char* file_prefix);

void vectors_to_tri_geometry(const std::vector<float>& vertices,
							 const std::vector<unsigned int>& indices,
							 const std::vector<float>& colors,
							 cvcraw_geometry::cvcgeom_t* geom);

void vectors_to_line_geometry(const std::vector<float>& vertices,
							  const std::vector<unsigned int>& indices,
							  float r, float g, float b, float a,
							  cvcraw_geometry::cvcgeom_t* geom);

void vectors_to_point_geometry(const std::vector<float>& vertices,
							   const std::vector<float>& colors,
							   cvcraw_geometry::cvcgeom_t* geom);
#endif
