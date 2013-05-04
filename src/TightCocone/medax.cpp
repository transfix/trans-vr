/*
  Copyright 2007-2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of TightCocone.

  TightCocone is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  TightCocone is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <TightCocone/medax.h>

namespace TightCocone
{

bool
is_VF_on_boundary_of_medax(const Triangulation& triang,
                           const Edge& e)
{
   Cell_handle c = e.first;
   int uid = e.second, vid = e.third;
   Facet_circulator fcirc = triang.incident_facets(e);
   Facet_circulator begin = fcirc;
   do{
      Cell_handle _c = (*fcirc).first;
      int _id = (*fcirc).second;
      int _uid = -1, _vid = -1;
      triang.has_vertex(_c, _id, c->vertex(uid), _uid);
      triang.has_vertex(_c, _id, c->vertex(vid), _vid);
      int _wid = 6-_id-_uid-_vid;
      if( !_c->VF_on_medax(_uid, _wid) &&
          !_c->VF_on_medax(_vid, _wid) )
         return true;
      fcirc++;
   } while( fcirc != begin );
   return false;
}

//-------------------------------------------------------------------
// compute_dia_of_tang_umbrella
//-------------
// Compute the biggest diameter of the triangles forming the tangent
// umbrella. These triangles must satisfy Cocone condition to be 
// included in the tangent umbrella.
//------------------------------------------------------------------
void 
compute_dia_of_tang_umbrella( const Triangulation &triang) 
{
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

    double cir_radius = sq_cr_tr_3( ch->vertex((id+1)%4)->point(),
                                    ch->vertex((id+2)%4)->point(),
                                    ch->vertex((id+3)%4)->point() );
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
// detect_axis_candidates 
//------------------
// Computes both inner and outer medial axis
//---------------------------------------------------------------------
void detect_axis_candidates( Triangulation &triang,
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
    if( ! is_inside_VF(triang, (*eit)) ) continue;
    // mark this edge (dual VF) as a possible candidate.
    Facet_circulator fcirc = triang.incident_facets((*eit));
    Facet_circulator begin_fcirc = fcirc;
    do{
       Cell_handle _c = (*fcirc).first; int _id = (*fcirc).second;
       int _vid = -1, _wid = -1;
       triang.has_vertex(_c, _id, vh, _vid);
       triang.has_vertex(_c, _id, wh, _wid);
       _c->e_tag[_vid][_wid] = true;
       _c->e_tag[_wid][_vid] = true;
       fcirc ++;
    }while(fcirc != begin_fcirc);
  }
  return;
}

// -- Using the conditions, compute a medial axis which is homotopy equivalent to
// the surface.
void
compute_axis(Triangulation& triang)
{
   // mark all the interior VFs to be on medax and then peel away
   // from boundary.
   for(FEI eit = triang.finite_edges_begin();
      eit != triang.finite_edges_end(); eit ++)
   {
      Cell_handle c = (*eit).first;
      int uid = (*eit).second, vid = (*eit).third;
      CGAL_assertion( ! c->VF_on_medax(uid,vid) );
      if( ! is_inside_VF(triang, (*eit)) ) continue;
      Facet_circulator fcirc = triang.incident_facets((*eit));
      Facet_circulator begin = fcirc;
      do{
         Cell_handle _c = (*fcirc).first;
         int _id = (*fcirc).second;
         int _uid = -1, _vid = -1;
         triang.has_vertex(_c, _id, c->vertex(uid), _uid);
         triang.has_vertex(_c, _id, c->vertex(vid), _vid);
         _c->set_VF_on_medax(_uid,_vid,true);
         fcirc++;
      } while(fcirc != begin);
   }

   // debug
   ofstream fout;
   fout.open("test");
   fout << "{LIST" << endl;
   for(FEI eit = triang.finite_edges_begin();
      eit != triang.finite_edges_end(); eit ++)
   {
      if( ! is_inside_VF(triang, (*eit)) ) continue;
      if( is_VF_on_boundary_of_medax(triang, (*eit)) )
         draw_VF(triang, (*eit), 0, 1, 0, 1, fout);
      else
      {
         if( (*eit).first->e_tag[(*eit).second][(*eit).third] )
            draw_VF(triang, (*eit), 1, 1, 0, 1, fout);
         else
            draw_VF(triang, (*eit), 1, 0, 0, 1, fout);
      }
   }
   fout << "}" << endl;
   fout.close();
   // end debug

   vector<Edge> peel;
   for(FEI eit = triang.finite_edges_begin();
      eit != triang.finite_edges_end(); eit ++)
   {
      if( ! is_inside_VF(triang, (*eit)) ) continue;
      Cell_handle c = (*eit).first;
      int uid = (*eit).second, vid = (*eit).third;
      
      if( c->e_visited[uid][vid] ) continue;
      // if its e_tag is true continue because then it is a true medial axis candidate.
      if( c->e_tag[uid][vid] ) continue;
      // otherwise put it in peel_stack if it is on the boundary.
      if( ! is_VF_on_boundary_of_medax(triang, (*eit)) ) continue;
      peel.push_back((*eit));
      Facet_circulator fcirc = triang.incident_facets((*eit));
      Facet_circulator begin = fcirc;
      do{
         Cell_handle _c = (*fcirc).first;
         int _id = (*fcirc).second;
         int _uid = -1, _vid = -1;
         triang.has_vertex(_c, _id, c->vertex(uid), _uid);
         triang.has_vertex(_c, _id, c->vertex(vid), _vid);
         CGAL_assertion( ! _c->e_tag[_uid][_vid] );
         _c->set_VF_on_medax(_uid,_vid,false);
         _c->e_visited[_uid][_vid] = true;
         fcirc++;
      } while(fcirc != begin);

      while( ! peel.empty() )
      {
         Edge e = peel.back();
         peel.pop_back();
         Cell_handle c = e.first;
         int uid = e.second, vid = e.third;

         // this VF must be visited and not on medax.
         Facet_circulator fcirc = triang.incident_facets(e);
         Facet_circulator begin = fcirc;
         do {
            Cell_handle _c = (*fcirc).first;
            int _id = (*fcirc).second;
            int _uid = -1, _vid = -1;
            triang.has_vertex(_c, _id, c->vertex(uid), _uid);
            triang.has_vertex(_c, _id, c->vertex(vid), _vid);
            CGAL_assertion(_c->e_visited[_uid][_vid]);
            CGAL_assertion(!_c->VF_on_medax(_uid, _vid) );
            CGAL_assertion(!_c->e_tag[_uid][_vid] );

            // (_c,_uid,_vid) is this VF.
            int _wid = 6 - _uid - _vid - _id;
            // other two VFs incident on the VE dual to (_c,_fid) are
            // (_c,_uid,_wid) and (_c,_vid,_wid).

            if( is_inside_VF(triang, Edge(_c,_uid,_wid)) &&
                ! _c->e_visited[_uid][_wid] &&
                ! _c->e_tag[_uid][_wid] &&
                is_VF_on_boundary_of_medax(triang, Edge(_c,_uid,_wid)) )
            {
               peel.push_back(Edge(_c,_uid,_wid));
               // mark the VF visited and not on medax.
               Facet_circulator fc = triang.incident_facets(Edge(_c,_uid,_wid));
               Facet_circulator beg = fc;
               do{
                  int tu = -1, tw = -1;
                  triang.has_vertex((*fc).first, (*fc).second, _c->vertex(_uid), tu);
                  triang.has_vertex((*fc).first, (*fc).second, _c->vertex(_wid), tw);
                  (*fc).first->e_visited[tu][tw] = true;
                  CGAL_assertion( ! (*fc).first->e_tag[tu][tw] );
                  (*fc).first->set_VF_on_medax(tu, tw, false);
                  fc++;
               } while(fc != beg);
            }
            if( is_inside_VF(triang, Edge(_c,_vid,_wid)) &&
                ! _c->e_visited[_vid][_wid] &&
                ! _c->e_tag[_vid][_wid] &&
                is_VF_on_boundary_of_medax(triang, Edge(_c,_vid,_wid)) )
            {
               peel.push_back(Edge(_c,_vid,_wid));
               Facet_circulator fc = triang.incident_facets(Edge(_c,_vid,_wid));
               Facet_circulator beg = fc;
               do{
                  int tv = -1, tw = -1;
                  triang.has_vertex((*fc).first, (*fc).second, _c->vertex(_vid), tv);
                  triang.has_vertex((*fc).first, (*fc).second, _c->vertex(_wid), tw);
                  (*fc).first->e_visited[tv][tw] = true;
                  CGAL_assertion( ! (*fc).first->e_tag[tv][tw] );
                  (*fc).first->set_VF_on_medax(tv, tw, false);
                  fc++;
               } while(fc != beg);
            }
            fcirc++;
         } while( fcirc != begin );
      }
   }
}

void
mark_connected_medial_axis_comp(Triangulation& triang, int& biggest_medax_comp_id)
{
   vector<Edge> connected_comp_walk;
   int comp_id = -1;
   int max_comp_size = -1;
   for(FEI eit = triang.finite_edges_begin();
      eit != triang.finite_edges_end(); eit ++)
   {
      Cell_handle c = (*eit).first;
      int uid = (*eit).second, vid = (*eit).third;
      if( ! is_inside_VF(triang, (*eit)) ) continue;
      if( ! c->VF_on_medax(uid, vid)) continue;
      if( c->medax_comp_id[uid][vid] != -1 ) continue;

      comp_id ++;
      int current_comp_size = 0;

      connected_comp_walk.push_back((*eit));
      Facet_circulator fcirc = triang.incident_facets((*eit));
      Facet_circulator begin = fcirc;
      do{
         Cell_handle _c = (*fcirc).first;
         int _id = (*fcirc).second;
         int _uid = -1, _vid = -1;
         triang.has_vertex(_c, _id, c->vertex(uid), _uid);
         triang.has_vertex(_c, _id, c->vertex(vid), _vid);
         CGAL_assertion( _c->VF_on_medax(_uid,_vid) );
         _c->medax_comp_id[_uid][_vid] = comp_id;
         _c->medax_comp_id[_vid][_uid] = comp_id;
         fcirc++;
      } while(fcirc != begin);

      while( ! connected_comp_walk.empty() )
      {
         Edge e = connected_comp_walk.back();
         connected_comp_walk.pop_back();
         current_comp_size ++;
         Cell_handle c = e.first;
         int uid = e.second, vid = e.third;
         // this VF must have comp_id != -1 and it must be on medax.
         Facet_circulator fcirc = triang.incident_facets(e);
         Facet_circulator begin = fcirc;
         do {
            Cell_handle _c = (*fcirc).first;
            int _id = (*fcirc).second;
            int _uid = -1, _vid = -1;
            triang.has_vertex(_c, _id, c->vertex(uid), _uid);
            triang.has_vertex(_c, _id, c->vertex(vid), _vid);
            CGAL_assertion(_c->VF_on_medax(_uid, _vid) );
            CGAL_assertion(_c->medax_comp_id[_uid][_vid] != -1);
         
            // (_c,_uid,_vid) is this VF.
            int _wid = 6 - _uid - _vid - _id;
            // other two VFs incident on the VE dual to (_c,_fid) are
            // (_c,_uid,_wid) and (_c,_vid,_wid).
            if( is_inside_VF(triang, Edge(_c,_uid,_wid)) &&
                _c->medax_comp_id[_uid][_wid] == -1 &&
                _c->VF_on_medax(_uid,_wid) )
            {
               connected_comp_walk.push_back(Edge(_c,_uid,_wid));
               // mark the comp_id of VF.
               Facet_circulator fc = triang.incident_facets(Edge(_c,_uid,_wid));
               Facet_circulator beg = fc;
               do{
                  int tu = -1, tw = -1;
                  triang.has_vertex((*fc).first, (*fc).second, _c->vertex(_uid), tu);
                  triang.has_vertex((*fc).first, (*fc).second, _c->vertex(_wid), tw);
                  (*fc).first->medax_comp_id[tu][tw] = comp_id;
                  (*fc).first->medax_comp_id[tw][tu] = comp_id;
                  fc++;
               } while(fc != beg);
            }
            if( is_inside_VF(triang, Edge(_c,_vid,_wid)) &&
                _c->medax_comp_id[_vid][_wid] == -1 &&
                _c->VF_on_medax(_vid,_wid) )
            {
               connected_comp_walk.push_back(Edge(_c,_vid,_wid));
               Facet_circulator fc = triang.incident_facets(Edge(_c,_vid,_wid));
               Facet_circulator beg = fc;
               do{
                  int tv = -1, tw = -1;
                  triang.has_vertex((*fc).first, (*fc).second, _c->vertex(_vid), tv);
                  triang.has_vertex((*fc).first, (*fc).second, _c->vertex(_wid), tw);
                  (*fc).first->medax_comp_id[tv][tw] = comp_id;
                  (*fc).first->medax_comp_id[tw][tv] = comp_id;
                  fc++;
               } while(fc != beg);
            }
            fcirc++;
         } while( fcirc != begin );
      }
      if( current_comp_size > max_comp_size )
      {
         max_comp_size = current_comp_size;
         biggest_medax_comp_id = comp_id;
      }
   }   
   cerr << "[" << biggest_medax_comp_id << ", " << max_comp_size << "]";
}

void
compute_medial_axis(Triangulation &triang,
		    const double theta, const double medial_ratio,
                    int& biggest_medax_comp_id)
{
  compute_dia_of_tang_umbrella( triang ); 
  cerr <<".";
  detect_axis_candidates( triang, theta, medial_ratio);
  cerr <<".";
  compute_axis( triang );
  cerr <<".";
  // optional. tag medial axis connected component id.
  mark_connected_medial_axis_comp( triang, biggest_medax_comp_id );
  cerr <<".";
  // end optional.

  return;
}

}
