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

#include <SecondaryStructures/init.h>
#include <SecondaryStructures/op.h>
#include <SecondaryStructures/robust_cc.h>

using namespace SecondaryStructures;

//a flag that will be set if the call to circumsphere fails
static bool cgal_failed;

//the handler that will flag the operation as having failed
//oh, what the heck, let's make it static while we're at it
static void failure_func(const char* type, const char* exp, const char* file, int line,	const char* expl)
{
	//bad CGAL!
	cgal_failed = true;
	return;
}

// Initialize some of the attributes of the triangulation.
void initialize(Triangulation& triang)
{
	// set vertex id.
	int id = 0;
	for (FVI vit = triang.finite_vertices_begin();
			vit != triang.finite_vertices_end(); vit ++)
	{
		vit->id = id++;
		vit->visited = false;
		vit->bad = false;
		vit->bad_neighbor = false;
	}
	// set cell id.
	id = 0;
	for (ACI cit = triang.all_cells_begin();
			cit != triang.all_cells_end(); cit ++)
	{
		cit->id = id++;
		cit->visited = false;
		cit->outside = false;
		cit->transp = false;
		for (int id = 0 ; id < 4; id++)
		{
			cit->set_cocone_flag(id,false);
			cit->neighbor(id)->set_cocone_flag(cit->neighbor(id)->index(cit),false);
			cit->bdy[id] = false;
			cit->opaque[id] = false;
			for (int k = 0; k < 4; k ++)
			{
				cit->umbrella_member[id][k] = -1;
			}
		}
		// set the convex hull points.
		if (! triang.is_infinite(cit))
		{
			continue;
		}
		for (int i = 0; i < 4; i ++)
		{
			if (! triang.is_infinite(cit->vertex(i)))
			{
				continue;
			}
			cit->vertex((i+1)%4)->set_convex_hull(true);
			cit->vertex((i+2)%4)->set_convex_hull(true);
			cit->vertex((i+3)%4)->set_convex_hull(true);
		}
	}

/*	cout<<"id = :" << id << endl;
	//zq test
	for(ACI cit = triang.all_cells_begin(); cit != triang.all_cells_end(); cit ++)
	{
	   for(int i=0; i< 4; i++)
	   	cout<<cit->vertex(i)->point() <<" |";
		cout << endl;
		if(cit->vertex(0)->point()==cit->vertex(3)->point()) {cout<<"wrong:"; }
	} */
		
	
}

// Compute all the finite voronoi vertices and the circumradius of all the finite cells.
void compute_voronoi_vertex_and_cell_radius(Triangulation& triang)
{
	bool is_there_any_problem_in_VV_computation = false;
	for (FCI cit = triang.finite_cells_begin();
			cit != triang.finite_cells_end(); cit ++)
	{
		//we tell CGAL to call our function if there is a problem
		//we also tell it not to die if things go haywire
		//CGAL::Failure_function old_ff = CGAL::set_error_handler(failure_func);
		//CGAL::Failure_behaviour old_fb = CGAL::set_error_behaviour(CGAL::CONTINUE);
		// be optimistic :-)
		//this is a global
		cgal_failed = false;
		cit->set_voronoi(triang.dual(cit));
		bool is_correct_computation = !cgal_failed;
		is_there_any_problem_in_VV_computation |= !is_correct_computation;
		if (cgal_failed)
		{
			// set cc the centroid of the cell.
			Vector cc = CGAL::NULL_VECTOR;
			for (int i = 0; i < 4; i ++)
			{
				cc = cc + (cit->vertex(i)->point() - CGAL::ORIGIN);
			}
			cc = (1./4.)*cc;
			cit->set_voronoi(CGAL::ORIGIN + cc);
		}
		//put everything back the way we found it,
		//CGAL::set_error_handler(old_ff);
		//CGAL::set_error_behaviour(old_fb);
		// set the cell radius.
		cit->set_cell_radius(CGAL::to_double((cit->vertex(0)->point()-cit->voronoi()) *(cit->vertex(0)->point()-cit->voronoi())));
	}
	return;
}
