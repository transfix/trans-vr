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

#include <TightCocone/op.h>

namespace TightCocone
{

extern vector<double> bounding_box;

void
draw_ray(const Ray& ray,
         const double& r, 
         const double& g, 
         const double& b, 
         const double& a, 
         ofstream& fout)
{
   fout << "{OFF" << endl;
   fout << "2 1 0" << endl;
   fout << ray.source() << endl;
   fout << (ray.source() - CGAL::ORIGIN) + ray.to_vector() << endl;
   fout << "2\t0 1 " << r << " " << g << " " << b << " " << a << endl;
   fout << "}" << endl;
}

void
draw_segment(const Segment& segment, 
             const double& r, 
             const double& g, 
             const double& b, 
             const double& a,
             ofstream& fout)
{
   fout << "{OFF" << endl;
   fout << "2 1 0" << endl;
   fout << segment.point(0) << endl;
   fout << segment.point(1) << endl;
   fout << "2\t0 1 " << r << " " << g << " " << b << " " << a << endl;
   fout << "}" << endl;
}


void
draw_poly(const vector<Point>& poly,
          const double& r, 
          const double& g, 
          const double& b, 
          const double& a,
          ofstream& fout)
{
   fout << "{OFF" << endl;
   fout << (int)poly.size() << " 1 0" << endl;
   for(int i = 0; i < (int)poly.size(); i ++)
      fout << poly[i] << endl;
   fout << (int)poly.size() << "\t";
   for(int i = 0; i < (int)poly.size(); i ++)
      fout << i << " ";
   fout << r << " " << g << " " << b << " " << a << endl;
   fout << "}" << endl;
}


void
draw_VF(const Triangulation& triang,
        const Edge& dual_e, 
        const double& r, 
        const double& g, 
        const double& b, 
        const double& a,
        ofstream& fout)
{
   Facet_circulator fcirc = triang.incident_facets(dual_e);
   Facet_circulator begin = fcirc;
   vector<Point> vvset;
   do
   {
      Cell_handle cc = (*fcirc).first;
      vvset.push_back(cc->voronoi());
      fcirc ++;
   } while(fcirc != begin);

   fout << "{OFF" << endl;
   fout << (int)vvset.size() << " 1 0" << endl;
   for(int i = 0; i < (int)vvset.size(); i ++)
      fout << vvset[i] << endl;
   fout << (int)vvset.size() << "\t";
   for(int i = 0; i < (int)vvset.size(); i ++)
      fout << i << " ";
   fout << r << " " << g << " " << b << " " << a << endl;
   fout << "}" << endl;
}

void
draw_tetra(const Cell_handle& cell,
           const double& r, 
           const double& g, 
           const double& b, 
           const double& a,
           ofstream& fout)
{
   fout << "{OFF" << endl;
   fout << "4 4 0" << endl;
   for(int i = 0; i < 4; i ++)
      fout << cell->vertex(i)->point() << endl;
   for(int i = 0; i < 4; i ++)
      fout << "3\t" << (i+1)%4 << " " 
                    << (i+2)%4 << " "
                    << (i+3)%4 << " "
                    << r << " " << g << " " << b << " " << a << endl;
   fout << "}" << endl;
}

void 
write_wt( const Triangulation &triang,
	  const char* file_prefix) 
{
  char filename[100];
  strcat(strcpy(filename, file_prefix), ".surf");
  ofstream fout;
  fout.open(filename);
  if(! fout)
  {
     cerr << "Can not open " << filename << " for writing. " << endl;
     exit(1);
  }
  // Count number of facets on the surface.
  int num_facets = 0;
  for ( FFI fit = triang.finite_facets_begin();
	fit != triang.finite_facets_end(); ++fit) 
     if((*fit).first->cocone_flag((*fit).second) )
        num_facets ++;

  fout <<"OFF" << endl;
  fout << triang.number_of_vertices()
       << " " << num_facets << " 0" << endl;

  // Write the vertices.
  for ( FVI vit = triang.finite_vertices_begin();
	vit != triang.finite_vertices_end(); ++vit) 
     fout << vit->point() << endl;
  // Write the facets.
  for ( FFI fit = triang.finite_facets_begin();
	fit != triang.finite_facets_end(); ++fit) 
  {
     Cell_handle c[2] = {(*fit).first, (*fit).first->neighbor((*fit).second)};
     int id[2] = {c[0]->index(c[1]), c[1]->index(c[0])};
     if(! c[0]->cocone_flag(id[0]) )
        continue;
     CGAL_assertion( c[0]->bdy[id[0]] && c[1]->bdy[id[1]] );
     CGAL_assertion( c[0]->outside != c[1]->outside );

     Vertex_handle vh[3] = { c[0]->vertex((id[0]+1)%4),
                             c[0]->vertex((id[0]+2)%4),
                             c[0]->vertex((id[0]+3)%4) };
     if( ! c[0]->outside )
        if( CGAL::is_negative( Tetrahedron( vh[0]->point(), vh[1]->point(), vh[2]->point(), 
                                            c[0]->vertex(id[0])->point() ).volume() ) )
           fout << "3\t" << vh[0]->id << " " << vh[1]->id << " " << vh[2]->id << " ";
        else
           fout << "3\t" << vh[1]->id << " " << vh[0]->id << " " << vh[2]->id << " ";
     else
        if( CGAL::is_negative( Tetrahedron( vh[0]->point(), vh[1]->point(), vh[2]->point(), 
                                            c[1]->vertex(id[1])->point() ).volume() ) )
           fout << "3\t" << vh[0]->id << " " << vh[1]->id << " " << vh[2]->id << " ";
        else
           fout << "3\t" << vh[1]->id << " " << vh[0]->id << " " << vh[2]->id << " ";
    fout << "1 1 1 0.3" << endl;
  }
  fout.close();
  return;
}


void 
write_iobdy( const Triangulation &triang,
	     const char* file_prefix) 
{

  char filename[100];
  strcat(strcpy(filename, file_prefix), ".io");

  ofstream fout;
  fout.open(filename);

  if(! fout)
  {
	  cerr << "Can not open " << filename << " for writing. " << endl;
	  exit(1);
  }

  // Count
  int num_facets = 0;
  for ( FFI fit = triang.finite_facets_begin();
	fit != triang.finite_facets_end(); ++fit) {
    Cell_handle ch = (*fit).first;
    int id = (*fit).second;
    if(ch->outside != ch->neighbor(id)->outside)
	    num_facets ++;
  }
  
  // The header of the output file

  fout <<"OFF";
  fout <<"  " << triang.number_of_vertices(); //The number of points
  fout <<" " << num_facets; //The number of facets
  fout <<" 0" << endl;

  // Write the vertices.
  for ( FVI vit = triang.finite_vertices_begin();
	vit != triang.finite_vertices_end(); ++vit) 
    fout << vit->point() << endl;
  
  for ( FFI fit = triang.finite_facets_begin();
	fit != triang.finite_facets_end(); ++fit) 
  {
    Cell_handle ch = (*fit).first;
    int id = (*fit).second;

    if(ch->outside == ch->neighbor(id)->outside)
	    continue;
    
    fout << " 3\t";
    for (int i = 1; i <= 3; i++)
    	fout << " " << ch->vertex((id+i)%4)->id;
    fout << "\t " <<  "1 1 1 1 \n"; ;
  }

  fout.close();
}


void 
write_axis( const Triangulation& triang,
            const int& biggest_medax_comp_id,
	    const char* file_prefix)
{
  char filename[100];
  strcat(strcpy(filename, file_prefix), ".ax");
  ofstream fout;
  fout.open(filename);
  if(! fout)
  {  cerr << "Can not open " << filename << " for writing. " << endl; exit(1); }

  char biggest_comp_filename[100];
  strcat(strcpy(biggest_comp_filename, file_prefix), ".00.ax");
  ofstream fout_biggest;
  fout_biggest.open(biggest_comp_filename);
  if(! fout_biggest )
  {  cerr << "Can not open " << biggest_comp_filename << " for writing. " << endl; exit(1); }

  fout << "{LIST" << endl;
  fout_biggest << "{LIST" << endl;
  for(FEI eit = triang.finite_edges_begin();
     eit != triang.finite_edges_end(); eit ++)
  {
     Cell_handle c = (*eit).first;
     int uid = (*eit).second, vid = (*eit).third;
     if( is_inf_VF(triang, c, uid, vid) ) continue; // inf VF
     if( ! is_inside_VF(triang, (*eit)) ) continue; // non-inside VF
     if( is_VF_outside_bounding_box( triang, (*eit), bounding_box) ) continue; // outside BBOX.
     if( ! c->VF_on_medax(uid, vid)) continue; // non-medax VF
     CGAL_assertion( c->medax_comp_id[uid][vid] != -1 ); // should be in some component.
     // choice of color.
     //srand48( c->medax_comp_id[uid][vid] );
     srand( c->medax_comp_id[uid][vid] );
     //double r = drand48(), g = drand48(), b = drand48(), a = 0;
     double 
       r = double(rand())/double(RAND_MAX), 
       g = double(rand())/double(RAND_MAX), 
       b = double(rand())/double(RAND_MAX), 
       a = 0;
     if( c->medax_comp_id[uid][vid] == biggest_medax_comp_id )
        a = 1;
     if( ! c->e_tag[uid][vid] )
     { r = 1; g = 0; b = 0; a = 0.3; }
     fout << "# " << c->medax_comp_id[uid][vid] << endl;
     draw_VF(triang, (*eit), r, g, b, a, fout);
     if( c->medax_comp_id[uid][vid] == biggest_medax_comp_id )
        draw_VF(triang, (*eit), r, g, b, a, fout_biggest);
  }
  fout << "}" << endl;
  fout_biggest << "}" << endl;
}

CVCGEOM_NAMESPACE::cvcgeom_t
wt_to_geometry( const Triangulation &triang)
{
  CVCGEOM_NAMESPACE::cvcgeom_t result;

  // Count number of facets on the surface.
  int num_facets = 0;
  for ( FFI fit = triang.finite_facets_begin();
	fit != triang.finite_facets_end(); ++fit) 
     if((*fit).first->cocone_flag((*fit).second) )
        num_facets ++;

  //collect verts into a temp vector
  {
    std::vector<Point> tmppts;
    for ( FVI vit = triang.finite_vertices_begin();
	  vit != triang.finite_vertices_end(); ++vit) 
      tmppts.push_back(vit->point());

    // Write the vertices.
    for ( std::vector<Point>::iterator p = tmppts.begin();
	  p != tmppts.end();
	  p++)
      {
	//fout << vit->point() << endl;
    CVCGEOM_NAMESPACE::cvcgeom_t::point_t newVertex;
	  newVertex[0] = p->x();
	  newVertex[1] = p->y();
	  newVertex[2] = p->z();
	  result.points().push_back(newVertex);
      }
  }
    
  // Write the facets.
  CVCGEOM_NAMESPACE::cvcgeom_t::triangle_t newTri;
  int cnt = 0;
  for ( FFI fit = triang.finite_facets_begin();
	fit != triang.finite_facets_end(); ++fit) 
  {
     Cell_handle c[2] = {(*fit).first, (*fit).first->neighbor((*fit).second)};
     int id[2] = {c[0]->index(c[1]), c[1]->index(c[0])};
     if(! c[0]->cocone_flag(id[0]) )
        continue;
     CGAL_assertion( c[0]->bdy[id[0]] && c[1]->bdy[id[1]] );
     CGAL_assertion( c[0]->outside != c[1]->outside );

     Vertex_handle vh[3] = { c[0]->vertex((id[0]+1)%4),
                             c[0]->vertex((id[0]+2)%4),
                             c[0]->vertex((id[0]+3)%4) };
     if( ! c[0]->outside )
        if( CGAL::is_negative( Tetrahedron( vh[0]->point(), vh[1]->point(), vh[2]->point(), 
                                            c[0]->vertex(id[0])->point() ).volume() ) )
	  {
	    //fout << "3\t" << vh[0]->id << " " << vh[1]->id << " " << vh[2]->id << " ";
		newTri[0] = vh[0]->id;
		newTri[1] = vh[1]->id;
		newTri[2] = vh[2]->id;
		result.triangles().push_back(newTri);
	  }
        else
	  {
	    //fout << "3\t" << vh[1]->id << " " << vh[0]->id << " " << vh[2]->id << " ";
		newTri[0] = vh[1]->id;
		newTri[1] = vh[0]->id;
		newTri[2] = vh[2]->id;
		result.triangles().push_back(newTri);

	  }
     else
        if( CGAL::is_negative( Tetrahedron( vh[0]->point(), vh[1]->point(), vh[2]->point(), 
                                            c[1]->vertex(id[1])->point() ).volume() ) )
	  {
	    //fout << "3\t" << vh[0]->id << " " << vh[1]->id << " " << vh[2]->id << " ";
		newTri[0] = vh[0]->id;
		newTri[1] = vh[1]->id;
		newTri[2] = vh[2]->id;
		result.triangles().push_back(newTri);
	  }
        else
	  {
	    //fout << "3\t" << vh[1]->id << " " << vh[0]->id << " " << vh[2]->id << " ";
		newTri[0] = vh[1]->id;
		newTri[1] = vh[0]->id;
		newTri[2] = vh[2]->id;
		result.triangles().push_back(newTri);

	  }
     //fout << "1 1 1 0.3" << endl;
     cnt++;
  }
  //fout.close();
  return result;
}


}
