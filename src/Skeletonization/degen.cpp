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

#include <Skeletonization/degen.h>

namespace Skeletonization
{

// *** This is not a general purpose degeneracy check routine ***.
// is_degenerate
// --------------
// Takes 
// a) a Cell c, 
// b) a Facet of c, f = (c,fid), 
// c) an edge of c, e = (c,uid,vid),
// d) a point: driver of VF (dual(e)), called d.
// Checks the following
// 1. If any of the VVs of dual(e) is infinite.
// 2. If dual of any two VVs in dual(e) are cospherical.
// 3. If the vector from driver (d) to VV = dual(c) is not
//    within the convex VF = dual(e).
bool
is_degenerate_VF(const Triangulation& triang, 
                 const Cell_handle& c,
                 const int& fid,
                 const int& uid,
                 const int& vid,
                 const Point& d,
                 const char* prefix)
{
   char degen_op_filename[200];
   strcat(strcpy(degen_op_filename, prefix), ".degen_VF");
 
   // an extra check - probably not needed.
   if(triang.is_infinite(c) ||
      triang.is_infinite(c->neighbor(fid)) ||
      triang.is_infinite(c->neighbor(6 - fid - uid - vid)) )
      return true;

   vector<Cell_handle> VF;
   Facet_circulator fcirc = triang.incident_facets(Edge(c,uid,vid));
   Facet_circulator begin = fcirc;
   do 
   {
      if(triang.is_infinite((*fcirc).first)) 
      {
         cerr << "< Inf VF >";
         return true; // by check-1 it is degenerate.
      }
      Cell_handle cur_c = (*fcirc).first;
      int cur_fid = (*fcirc).second;
      // check if cur_c and its cur_fid neighbors are cospherical. 
      if( is_cospherical_pair(triang, Facet(cur_c,cur_fid)) )
      {
         cerr << "< Cosph VF >";
         return true; // by check-2 it is degenerate.
      }
      fcirc ++;
   } while(fcirc != begin);
   
   // check-3 
   Point vv[3];
   vv[0] = c->voronoi();
   vv[1] = c->neighbor(fid)->voronoi();
   vv[2] = c->neighbor(6 - fid - uid - vid)->voronoi();
   Vector v[3];
   v[0] = vv[0] - d;
   v[1] = vv[1] - d;
   v[2] = vv[2] - d;
   Vector v1xv0 = CGAL::cross_product(v[1], v[0]);
   Vector v0xv2 = CGAL::cross_product(v[0], v[2]);
   if(CGAL::to_double(v1xv0 * v0xv2) < 0)
   {
      ofstream fout;
      fout.open(degen_op_filename, ofstream::app);
      fout << "# prob : v1xv0 * v0xv2 = " << 
              CGAL::to_double(v1xv0 * v0xv2) << endl;
      fout << "{LIST " << endl;
      fout << "# VF - color yellow " << endl;
      draw_VF(triang, Edge(c, uid, vid), 1, 1, 0, 1, fout);
      fout << "# v0 : segment(driver, voronoi(c)) - color red " << endl;
      draw_segment(Segment(d, vv[0]), 1, 0, 0, 1, fout);
      fout << "# v1 : segment(driver, voronoi(c->neighbor1)) - color green " << endl;
      draw_segment(Segment(d, vv[1]), 0, 1, 0, 1, fout);
      fout << "# v2 : segment(driver, voronoi(c->neighbor2)) - color blue " << endl;
      draw_segment(Segment(d, vv[2]), 0, 0, 1, 1, fout);
      fout << "}" << endl;
      fout.close();

      cerr << "< - v1xv0 * v0xv2 < 0 - >";
      return true;
   }
   return false;
}

}
