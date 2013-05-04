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

#ifndef __HELIX_H__
#define __HELIX_H__

#include <SecondaryStructures/datastruct_ss.h>

using namespace SecondaryStructures;

typedef pair< vector<SecondaryStructures::Point>, SecondaryStructures::Vector> Cylinder;

class CVertex
{
	public:
		CVertex()
		{
		}
		CVertex(const SecondaryStructures::Point& p)
		{
			pos = p;
		}
		SecondaryStructures::Point pos;
		int id;
		bool visited;
		vector<int> inc_vid_list;
};

class CEdge
{
	public:
		CEdge()
		{
		}
		CEdge(const int& v1, const int& v2)
		{
			ep[0] = v1;
			ep[1] = v2;
		}
		int ep[2];
};

class Curve
{
	public:
		Curve()
		{
		}
		vector<CVertex> vert_list;
		vector<CEdge> edge_list;
};

#endif
