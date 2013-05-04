/*
  Copyright 2003 The University of Texas at Austin

        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of contourtree.

  contourtree is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  contourtree is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef COMPUTE_CT_H
#define COMPUTE_CT_H

typedef struct _CTVTX {
	float norm_x;
	float func_val;
} CTVTX;

typedef struct _CTEDGE {
	int v1;
	int v2;
} CTEDGE;

// INPUT :
//		uchar_data : unsigned char* type array for volumetric data
//		dim : dim[0]:x-direction , dim[1]:y-direction , dim[2]:z-direction dimension
// OUTPUT :
//		no_vtx : #vertices of contour tree
//		vtx_list : normalized position of x-coordinate (norm_x) and corresponding function value which can be
//		           acted as y-coordinate
//		edge_list : an array of edges composed of indices of two end vertices on each edge.
//
//	the memory of vtx_list and edge_list should not be allocated before calling this function
//  after calling this function, vtx_list and edge_list should be deallocated when it is not necessary.
//  JW note: deallocate with free()
void computeCT(unsigned char* uchar_data, int* dim, int& no_vtx, int& no_edge, CTVTX** vtx_list, CTEDGE** edge_list);

#endif

