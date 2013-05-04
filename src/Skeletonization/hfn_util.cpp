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

#include <Skeletonization/hfn_util.h>

namespace Skeletonization
{

bool
is_maxima(const Cell_handle& c)
{
   return (! is_outflow(Facet(c,0)) &&
           ! is_outflow(Facet(c,1)) &&
           ! is_outflow(Facet(c,2)) &&
           ! is_outflow(Facet(c,3)) );
}

bool
is_outflow(const Facet& f)
{
   Cell_handle c = f.first;
   int id = f.second;
   Point VV = c->voronoi();
   Point p[4] = {c->vertex((id+1)%4)->point(),
                 c->vertex((id+2)%4)->point(),
                 c->vertex((id+3)%4)->point(),
                 c->vertex(id)->point()};

   return (CGAL::to_double((Tetrahedron(p[0], p[1], p[2], p[3]).volume()) *
                           (Tetrahedron(p[0], p[1], p[2], VV).volume())) < 0);
}

bool
is_transversal_flow(const Facet& f)
{
   Cell_handle c = f.first;
   int id = f.second;

   Point p[3];
   for(int i = 0; i < 3; i ++)
      p[i] = c->vertex((id+i+1)%4)->point();
   for(int i = 0; i < 3; i ++)
   {
      Vector v0 = p[(i+1)%3] - p[i];
      Vector v1 = p[(i+2)%3] - p[i];

      if(CGAL::to_double(v0 * v1) < 0)
         return true;
   }
   return false;
}

bool
find_acceptor(const Cell_handle& c, const int& id,
              int& uid, int& vid, int& wid)
{
   if( ! is_transversal_flow(Facet(c,id))) return false;
   Point p[3] = {c->vertex((id+1)%4)->point(),
                 c->vertex((id+2)%4)->point(),
                 c->vertex((id+3)%4)->point()};
   for(int i = 0; i < 3; i ++)
      if( is_obtuse(p[(i+1)%3], p[(i+2)%3], p[i]) )
         wid = (id+i+1)%4;
   vertex_indices(id, wid, uid, vid);
   return true;
}

bool
is_i2_saddle(const Facet& f)
{
   Cell_handle c[2]; int id[2];
   c[0] = f.first; id[0] = f.second;
   c[1] = c[0]->neighbor(id[0]); id[1] = c[1]->index(c[0]);

   Point p[3];
   p[0] = c[0]->vertex((id[0]+1)%4)->point();
   p[1] = c[0]->vertex((id[0]+2)%4)->point();
   p[2] = c[0]->vertex((id[0]+3)%4)->point();
   Tetrahedron t[2];
   t[0] = Tetrahedron(p[0], p[1], p[2], c[0]->voronoi());
   t[1] = Tetrahedron(p[0], p[1], p[2], c[1]->voronoi());

   if( CGAL::to_double(t[0].volume()*t[1].volume()) >= 0) return false;
   for(int i = 0; i < 3; i ++)
   {
      Vector v1 = c[0]->vertex((id[0]+(i+1)%3+1)%4)->point() - 
                  c[0]->vertex((id[0]+i+1)%4)->point();
      Vector v2 = c[0]->vertex((id[0]+(i+2)%3+1)%4)->point() - 
	          c[0]->vertex((id[0]+i+1)%4)->point();
      if(cosine(v1,v2) < 0)
         return false;
   }
   return true;
}

bool is_i1_saddle(const Edge& e, const Triangulation& triang)
{
   // create the VF from e and triang.
   vector<Point> VF;
   Facet_circulator fcirc = triang.incident_facets(e);
   Facet_circulator begin = fcirc;
   do 
   {
      if(triang.is_infinite((*fcirc).first)) 
         return false;
      VF.push_back((*fcirc).first->voronoi());
      fcirc ++;
   } while(fcirc != begin);

   return does_intersect_convex_polygon_segment_3_in_3d(VF, 
                                                        Segment(e.first->vertex(e.second)->point(), 
                                                                e.first->vertex(e.third)->point()) );
}

bool
is_acceptor_for_any_VE(const Triangulation& triang, const Edge& e)
{
   Cell_handle cell = e.first;
   int uid = e.second, vid = e.third;

   Facet_circulator fcirc = triang.incident_facets(e);
   Facet_circulator begin = fcirc;
   do{
      Cell_handle c[2]; int id[2];
      c[0] = (*fcirc).first; id[0] = (*fcirc).second;
      c[1] = c[0]->neighbor(id[0]); id[1] = c[1]->index(c[0]);
      // VE is between c[0]->voronoi() and c[1]->voronoi().
      // dual is f(c[0], id[0]) = f(c[1], id[1]).
      if( ! is_transversal_flow(Facet(c[0], id[0])) )
      {
         CGAL_assertion( ! is_transversal_flow(Facet(c[1], id[1])));
         fcirc ++;
         continue;
      }
      // find if this VF is acceptor for this VE.
      int cur_uid = -1, cur_vid = -1, cur_wid = -1;
      if( find_acceptor(c[0], id[0], cur_uid, cur_vid, cur_wid) )
      {
         if((c[0]->vertex(cur_uid)->id == cell->vertex(uid)->id &&
             c[0]->vertex(cur_vid)->id == cell->vertex(vid)->id) ||
            (c[0]->vertex(cur_uid)->id == cell->vertex(vid)->id &&
             c[0]->vertex(cur_vid)->id == cell->vertex(uid)->id) )
            return true;
      }
      fcirc ++;
   } while(fcirc != begin);
   return false;
}

void
grow_maxima(Triangulation& triang, Cell_handle c_max)
{
   // mark it visited.
   c_max->visited = true;

   // Now grow the maximum through the other tetrahedra.
   vector<Facet> bdy_stack;

   for(int i = 0; i < 4; i ++)
   {
      bdy_stack.push_back(Facet(c_max, i));
      c_max->vertex(i)->set_pl(c_max->is_pl());
      c_max->vertex(i)->set_cyl(c_max->is_cyl());
   }

   while(! bdy_stack.empty())
   {
      Cell_handle old_c = bdy_stack.back().first;
      int old_id = bdy_stack.back().second;
      bdy_stack.pop_back();
      CGAL_assertion(old_c->visited);
      CGAL_assertion( ! old_c->outside );

      Cell_handle new_c = old_c->neighbor(old_id);
      int new_id = new_c->index(old_c);

     // If the new_c is infinite then no point in checking
     // the flow.
     if(triang.is_infinite(new_c))
        continue;
     // if the flow hits the boundary of the inner balls
     // and outer balls continue.
     if(new_c->outside)
        continue;

     // If new_c is already visited continue.
     if(new_c->visited) 
        continue;

     // if the flow is undefined then do the following.
     if(!((old_c->source(old_id) && new_c->terminus(new_id)) || 
	  (old_c->terminus(old_id) && new_c->source(new_id)) || 
	  (old_c->terminus(old_id) && new_c->terminus(new_id)) ) )
     {
        // they must be a cospherical pair.
	CGAL_assertion(old_c->cosph_pair(old_id) &&
		       new_c->cosph_pair(new_id));
        // if they are cospherical pair then this is
        // also true.
        CGAL_assertion(! old_c->source(old_id) &&
		       ! old_c->terminus(old_id) &&
		       ! new_c->source(new_id) &&
		       ! new_c->terminus(new_id) );
			    
       // some safety checks.
       // it can not be a maximum.
       CGAL_assertion(! is_maxima(new_c));
       // It can not be an infinite tetrahedron.
       CGAL_assertion(! triang.is_infinite(new_c));
       // new_c can't be already visited.
       CGAL_assertion(!new_c->visited);

       // take the new cell into the cluster.
       // mark it visited.
       new_c->visited = true;
       // propagate the pl-cyl marking to the vertices.
       for(int i = 0; i < 4; i ++)
       {
          new_c->vertex(i)->set_pl(c_max->is_pl());
          new_c->vertex(i)->set_cyl(c_max->is_cyl());
       }
       // get the new cells.
       for(int i = 1; i < 4; i ++)
       {
          if(new_c->neighbor((new_id+i)%4)->visited) 
	     continue;
	  bdy_stack.push_back(Facet(new_c, (new_id+i)%4));
       }
       continue;
     }
  
     // now there are three possibilities left for
     // this flow which is defined.
     if(old_c->source(old_id))
     {
        CGAL_assertion(new_c->terminus(new_id));
        continue;
     }
     if(old_c->terminus(old_id) &&
        new_c->terminus(new_id) )
     {
        // this is a saddle face.
        continue;
     }
     CGAL_assertion(old_c->terminus(old_id) &&
	            new_c->source(new_id) );
		    
     // some safety checks.
     // it can not be a maximum.
     CGAL_assertion(! is_maxima(new_c));
     // It can not be an infinite tetrahedron.
     CGAL_assertion(! triang.is_infinite(new_c));
     // new_c can't be already visited.
     CGAL_assertion(!new_c->visited);

     // mark it visited.
     new_c->visited = true;

     // propagate the pl-cyl marking to the vertices.
     for(int i = 0; i < 4; i ++)
     {
        new_c->vertex(i)->set_pl(c_max->is_pl());
        new_c->vertex(i)->set_cyl(c_max->is_cyl());
     }
     // get the new cells.
     for(int i = 1; i < 4; i ++)
     {
        if(new_c->neighbor((new_id+i)%4)->visited) 
	   continue;
	bdy_stack.push_back(Facet(new_c, (new_id+i)%4));
     }
   }
}


// ------------------------------------------------------
// find_flow_direction
// ------------------------
// Given a tetrahedralization find out the direction of flow 
// on each voronoi edge 
// ------------------------------------------------------
void
find_flow_direction(Triangulation &triang )
{


   for(FFI fit = triang.finite_facets_begin();
	fit != triang.finite_facets_end(); fit ++)
   {
	   Cell_handle c[2];
	   int id[2];

	   c[0] = (*fit).first; id[0] = (*fit).second;
	   c[1] = c[0]->neighbor(id[0]); id[1] = c[1]->index(c[0]);

	   if(triang.is_infinite(c[0]) || triang.is_infinite(c[1]))
	   {
		   // let c[0] be the finite one.
		   if(triang.is_infinite(c[0]))
		   {
			   Cell_handle temp = c[0];
			   c[0] = c[1];
			   c[1] = temp;
			   id[0] = c[0]->index(c[1]);
			   id[1] = c[1]->index(c[0]);
		   }

		   if(c[0]->dirty())
		   {
			   c[0]->set_source(id[0], true);
			   c[1]->set_terminus(id[1], true);
			   continue;
		   }

		   Tetrahedron tf = Tetrahedron(c[0]->vertex((id[0]+1)%4)->point(),
					   	c[0]->vertex((id[0]+2)%4)->point(),
						c[0]->vertex((id[0]+3)%4)->point(),
						c[0]->vertex(id[0])->point());
		   Tetrahedron tf_vp = Tetrahedron(c[0]->vertex((id[0]+1)%4)->point(),
				   		   c[0]->vertex((id[0]+2)%4)->point(),
						   c[0]->vertex((id[0]+3)%4)->point(),
						   c[0]->voronoi());

		   // check the flow here.
		   /* if(tf.volume()*tf_vp.volume() >= NT(0)) // should it be >= or >
		   */

		   if(! CGAL::is_negative(tf.volume()*tf_vp.volume())) // should it be >= or >
		   {
			   c[0]->set_terminus(id[0], true);
			   c[1]->set_terminus(id[1], true);
		   }
		   else
		   {
			   c[0]->set_source(id[0], true);
			   c[1]->set_terminus(id[1], true);
		   }

		   continue;
	   }
	   
	   // If the pair of tetrahedra are cospherical
	   // continue because the flow is undefined.
	   if(c[0]->cosph_pair(id[0])) continue;

	   if(c[0]->dirty() && !c[1]->dirty())
	   {
		   Cell_handle temp = c[0];
		   c[0] = c[1];
		   c[1] = temp;
		   id[0] = c[0]->index(c[1]);
		   id[1] = c[1]->index(c[0]);
	   }


	   Point p[5];
	   p[0] = c[0]->vertex(id[0])->point();
	   p[1] = c[0]->vertex((id[0]+1)%4)->point();
	   p[2] = c[0]->vertex((id[0]+2)%4)->point();
	   p[3] = c[0]->vertex((id[0]+3)%4)->point();
	   p[4] = c[1]->vertex(id[1])->point();
	   
	   Point vp[2];
	   vp[0] = c[0]->voronoi();
	   vp[1] = c[1]->voronoi();
	   
	   Tetrahedron t[2];
	   t[0] = Tetrahedron(p[1], p[2], p[3], p[0]);
	   t[1] = Tetrahedron(p[1], p[2], p[3], p[4]);
	   Tetrahedron tvp[2];
	   tvp[0] = Tetrahedron(p[1],p[2],p[3],vp[0]);
	   tvp[1] = Tetrahedron(p[1],p[2],p[3],vp[1]);

	   
	   // both the tetrahedra are finite.
	   
	   // case - 1 : none of them is dirty.
	   // in that case the direction of flow
	   // should be found correctly.
	   if(! c[0]->dirty() && ! c[1]->dirty())
	   {
		   // find out the flow relation between 
		   // the pair of tetrahedra.
		   
		   /* if(t[0].volume()*tvp[0].volume() < NT(0))
		   */
		   if(CGAL::is_negative(t[0].volume()*tvp[0].volume()))
		   {
			   //CGAL_assertion(t[1].volume()*tvp[1].volume() > NT(0));
			   // CGAL_assertion(CGAL::is_positive(t[1].volume()*tvp[1].volume()));
			   c[0]->set_source(id[0], true);
			   c[1]->set_terminus(id[1], true);
			   continue;
		   }
		   
		   /* if(t[1].volume()*tvp[1].volume() < NT(0))
		   */
		   if(CGAL::is_negative(t[1].volume()*tvp[1].volume()))
		   {
			   //CGAL_assertion(t[0].volume()*tvp[0].volume() > NT(0));
			   CGAL_assertion(CGAL::is_positive(t[0].volume()*tvp[0].volume()));
			   c[1]->set_source(id[1], true);
			   c[0]->set_terminus(id[0], true);
			   continue;
		   }
		   
		   /*if(t[0].volume()*tvp[0].volume() == NT(0))
		   */
		   if(CGAL::is_zero(t[0].volume()*tvp[0].volume()))
		   {
			   CGAL_assertion(! CGAL::is_negative(t[1].volume()*tvp[1].volume()));
			   c[0]->set_source(id[0], true);
			   c[1]->set_terminus(id[1], true);
			   continue;
		   }
		   
		   /* if(t[1].volume()*tvp[1].volume() == NT(0))
		   */
		   if(CGAL::is_zero(t[1].volume()*tvp[1].volume()))
		   {
			   //CGAL_assertion(t[0].volume()*tvp[0].volume() >= NT(0));
			   CGAL_assertion( ! CGAL::is_negative(t[0].volume()*tvp[0].volume()));
			   c[1]->set_source(id[1], true);
			   c[0]->set_terminus(id[0], true);
			   continue;
		   }

		   /* if(t[0].volume()*tvp[0].volume() > NT(0) &&
		   *    t[1].volume()*tvp[1].volume() > NT(0))
		   */
		   if(CGAL::is_positive(t[0].volume()*tvp[0].volume()) &&
		      CGAL::is_positive(t[1].volume()*tvp[1].volume()) )
		   {
			   c[0]->set_terminus(id[0], true);
			   c[1]->set_terminus(id[1], true);
			   continue;
		   }

		   continue;
	   }

	   // case 2 : both are dirty. 
	   // assign the direction arbitrarily.
	   if(c[0]->dirty() && c[1]->dirty())
	   {
		   c[0]->set_source(id[0], true);
		   c[0]->set_terminus(id[0], false);
		   c[1]->set_source(id[1], false);
		   c[1]->set_terminus(id[1], true);

		   continue;
	   }

	   // case - 3 : one of them is dirty.
	   // in that case the direction of flow
	   // is identified by the good tetrahedron.

	   CGAL_assertion(! c[0]->dirty() && c[1]->dirty());


	   /*if(t[0].volume()*tvp[0].volume() <= NT(0))
	   */
	   if(! CGAL::is_positive(t[0].volume()*tvp[0].volume()))
	   {
		   c[0]->set_source(id[0], true);
		   c[1]->set_terminus(id[1], true);
	   }
	   else
	   {
		   c[1]->set_source(id[1], true);
		   c[0]->set_terminus(id[0], true);
	   }
   }

}

}
