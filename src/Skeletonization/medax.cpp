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

#include <Skeletonization/medax.h>

namespace Skeletonization
{

  //-------------------------------------------------------------------
  //tang_umbrella
  //-------------
  //Choose the dual triangles of Vorinoi edges which are intersected
  //by the tangent polygon (refer to the medial paper).
  //------------------------------------------------------------------
  void 
  tang_umbrella( const Triangulation &triang) {

    // Check for the co-cone condition for every finite facet. 
    for ( FFI fit = triang.finite_facets_begin();
	  fit != triang.finite_facets_end(); ++fit) {
      Cell_handle ch = (*fit).first;
      int id = (*fit).second;
    
      // A finite facet must be incident to at least one finite cell.
      if ( triang.is_infinite( ch)) {
	Cell_handle c = ch;
	ch = ch->neighbor(id);
	id = ch->index( c);
      }
      CGAL_assertion( ! triang.is_infinite( ch));

      // Compute the radius of the circumcircle of the triangle.
      double t = CGAL::to_double((ch->vertex((id+1)%4)->point() - ch->voronoi())*
				 (ch->neighbor(id)->voronoi()- ch->voronoi()))/
	CGAL::to_double((ch->voronoi() - ch->neighbor(id)->voronoi())*
			(ch->voronoi() - ch->neighbor(id)->voronoi()));
    
      double cir_radius = CGAL::to_double((ch->voronoi() + 
					   t*(ch->neighbor(id)->voronoi() - 
					      ch->voronoi()) -
					   ch->vertex((id+1)%4)->point())*
					  (ch->voronoi() + 
					   t*(ch->neighbor(id)->voronoi() - 
					      ch->voronoi()) -
					   ch->vertex((id+1)%4)->point()));
					 
      // Iterate over the vertices incident to the facet.
      for ( int i = 1; i < 4; ++i) {
	Vertex_handle vh = ch->vertex((id+i)%4);
	double cos1 = cosine( ch->voronoi() - vh->point(), vh->normal());
	double cos2 = ( triang.is_infinite( ch->neighbor(id))) ? 
	  1.0 : cosine( ch->neighbor(id)->voronoi() - vh->point(), 
			vh->normal());
	if ( cos1 > cos2) 
	  swap( cos1, cos2);
	// Test for disjoint intervals [cos1, cos2] and [min_cos, max_cos].

	if ( (cos1 < 0.0) && (cos2 > 0.0)){
	  if(!vh->is_flat()) continue;
	  // Add the normals to the normal_stack
	  Vector facet_normal = 
	    CGAL::cross_product( ch->vertex((id+2)%4)->point() - 
				 ch->vertex((id+1)%4)->point(), 
				 ch->vertex((id+3)%4)->point() - 
				 ch->vertex((id+1)%4)->point() );
	  vh->normal_stack.push_back(facet_normal);
	
	  // Update diameter
	  if(vh->diameter() < cir_radius)
	    vh->set_diameter(cir_radius);
	}
      }
    } 
  }

  //--------------------------------------------------------------------
  // compute_axis 
  //------------------
  // Only compute the inner part of the medial axis
  //---------------------------------------------------------------------
  void compute_axis( Triangulation &triang,
		     const double theta, const double ratio)
  {
    double COS_MIN = cos(M_PI/2.0 + theta);
    double COS_MAX = cos(M_PI/2.0 - theta);
  
    // Count the number of the chosen Voronoi facets
    // and put the edge into the stack.
    for ( FEI eit = triang.finite_edges_begin();
	  eit != triang.finite_edges_end(); ++eit) {
      Cell_handle ch = (*eit).first;
      Vertex_handle vh = ch->vertex( (*eit).second);
      Vertex_handle wh = ch->vertex( (*eit).third);
    
      if(!vh->is_flat() && !wh->is_flat()) 
	continue;

      //choose by the angle
      Vector ev = wh->point() - vh->point();    
      Vector ew = vh->point() - wh->point();
      double vcos;
      bool vchoose = true;
      for(unsigned int i = 0; i < vh->normal_stack.size(); i++){
	vcos = cosine(vh->normal_stack[i], ev);
	if( (vcos < COS_MAX) && (vcos > COS_MIN))
	  vchoose = false;
      }
      double wcos;
      bool wchoose = true;
      for(unsigned int i = 0; i < wh->normal_stack.size(); i++){
	wcos = cosine(wh->normal_stack[i], ew);
	if( (wcos < COS_MAX) && (wcos > COS_MIN))
	  wchoose = false;
      }
 
      if(!vchoose || !wchoose){
	//choose by the ratio
	double length = CGAL::to_double((vh->point() - wh->point())*
					(vh->point() - wh->point()));
	if(!((length/vh->diameter() > ratio) &&
	     (length/wh->diameter() > ratio)))
	  continue;
      }
          
      /*
       * Currently we want to mark both inside and outside medial axis
       * and so I comment out the next code.
       *
       Cell_circulator begin = triang.incident_cells(*eit);
       Cell_circulator ccirc = begin;  
       // If there are infinite, non-inside cells or cells which are outside 
       // the clipbox, then discard the Vornoi facet.
       bool choose = true;
       do{
       if(triang.is_infinite(ccirc)){
       choose = false;
       break;
       }
       if(ccirc->outside){
       choose = false;
       break;
       }
       ccirc++;
       } while(ccirc != begin);
       if(!choose) continue;
      */

      Vertex_handle u = (*eit).first->vertex((*eit).second);
      Vertex_handle v = (*eit).first->vertex((*eit).third);
      Facet_circulator fcirc = triang.incident_facets((*eit));
      Facet_circulator begin_fcirc = fcirc;
      do{
	Cell_handle ch = (*fcirc).first; int fid = (*fcirc).second;
	int u_id = -1, v_id = -1;
	CGAL_assertion(triang.has_vertex(ch, fid, u, u_id));
	CGAL_assertion(u_id != -1);
	CGAL_assertion(triang.has_vertex(ch, fid, v, v_id));
	CGAL_assertion(v_id != -1);
	CGAL_assertion(u_id != v_id);

	ch->set_VF_on_medax(u_id,v_id,true);
	ch->set_VV_on_medax(true);

	fcirc ++;
      }while(fcirc != begin_fcirc);
    }
  }



  void
  compute_medial_axis(Triangulation &triang,
		      const double theta, const double medial_ratio)
  {
    tang_umbrella( triang);
    cerr <<".";
   
    compute_axis( triang, theta, medial_ratio);
    cerr <<"." << flush;
  }

}
