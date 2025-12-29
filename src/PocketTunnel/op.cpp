/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of PocketTunnel.

  PocketTunnel is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  PocketTunnel is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#include <PocketTunnel/op.h>

namespace PocketTunnel {

void draw_ray(const cgal_Ray &ray, const double &r, const double &g,
              const double &b, const double &a, ofstream &fout) {
  fout << "{OFF" << endl;
  fout << "2 1 0" << endl;
  fout << ray.source() << endl;
  fout << (ray.source() - CGAL::ORIGIN) + ray.to_vector() << endl;
  fout << "2\t0 1 " << r << " " << g << " " << b << " " << a << endl;
  fout << "}" << endl;
}

void draw_segment(const Segment &segment, const double &r, const double &g,
                  const double &b, const double &a, ofstream &fout) {
  fout << "{OFF" << endl;
  fout << "2 1 0" << endl;
  fout << segment.point(0) << endl;
  fout << segment.point(1) << endl;
  fout << "2\t0 1 " << r << " " << g << " " << b << " " << a << endl;
  fout << "}" << endl;
}

void draw_poly(const vector<Point> &poly, const double &r, const double &g,
               const double &b, const double &a, ofstream &fout) {
  fout << "{OFF" << endl;
  fout << (int)poly.size() << " 1 0" << endl;
  for (int i = 0; i < (int)poly.size(); i++)
    fout << poly[i] << endl;
  fout << (int)poly.size() << "\t";
  for (int i = 0; i < (int)poly.size(); i++)
    fout << i << " ";
  fout << r << " " << g << " " << b << " " << a << endl;
  fout << "}" << endl;
}

void draw_VF(const Triangulation &triang, const Edge &dual_e, const double &r,
             const double &g, const double &b, const double &a,
             ofstream &fout) {
  Facet_circulator fcirc = triang.incident_facets(dual_e);
  Facet_circulator begin = fcirc;
  vector<Point> vvset;
  do {
    Cell_handle cc = (*fcirc).first;
    vvset.push_back(cc->voronoi());
    fcirc++;
  } while (fcirc != begin);

  fout << "{OFF" << endl;
  fout << (int)vvset.size() << " 1 0" << endl;
  for (int i = 0; i < (int)vvset.size(); i++)
    fout << vvset[i] << endl;
  fout << (int)vvset.size() << "\t";
  for (int i = 0; i < (int)vvset.size(); i++)
    fout << i << " ";
  fout << r << " " << g << " " << b << " " << a << endl;
  fout << "}" << endl;
}

void draw_tetra(const Cell_handle &cell, const double &r, const double &g,
                const double &b, const double &a, ofstream &fout) {
  fout << "{OFF" << endl;
  fout << "4 4 0" << endl;
  for (int i = 0; i < 4; i++)
    fout << cell->vertex(i)->point() << endl;
  for (int i = 0; i < 4; i++)
    fout << "3\t" << (i + 1) % 4 << " " << (i + 2) % 4 << " " << (i + 3) % 4
         << " " << r << " " << g << " " << b << " " << a << endl;
  fout << "}" << endl;
}

//-----------------
// write_watertight
//-----------------
// Write out the boundary between inside and outside tetrehedra as surface.
//-----------------------------------------------------------------------
void write_wt(const Triangulation &triang, const char *file_prefix) {

  char filename[100];
  strcat(strcpy(filename, file_prefix), ".surf");

  ofstream fout;
  fout.open(filename);

  if (!fout) {
    cerr << "Can not open " << filename << " for writing. " << endl;
    exit(1);
  }

  // Count
  int num_facets = 0;
  for (FFI fit = triang.finite_facets_begin();
       fit != triang.finite_facets_end(); ++fit) {
    Cell_handle ch = (*fit).first;
    int id = (*fit).second;
    CGAL_assertion(ch->bdy[id] ==
                   ch->neighbor(id)->bdy[ch->neighbor(id)->index(ch)]);
    if (ch->cocone_flag(id))
      num_facets++;
  }

  // The header of the output file

  fout << "OFF" << endl;
  fout << triang.number_of_vertices(); // The number of points
  fout << " " << num_facets;           // The number of facets
  fout << " 0" << endl;

  // Write the vertices.
  for (FVI vit = triang.finite_vertices_begin();
       vit != triang.finite_vertices_end(); ++vit)
    fout << vit->point() << endl;

  for (FFI fit = triang.finite_facets_begin();
       fit != triang.finite_facets_end(); ++fit) {
    Cell_handle ch = (*fit).first;
    int id = (*fit).second;

    if (!ch->cocone_flag(id))
      continue;
    CGAL_assertion(ch->bdy[id] &&
                   ch->neighbor(id)->bdy[ch->neighbor(id)->index(ch)]);

    fout << " 3\t";
    for (int i = 1; i <= 3; i++)
      fout << " " << ch->vertex((id + i) % 4)->id;
    fout << "\t " << "1 1 1 0.3 \n";
    ;
  }

  fout.close();
}

void write_handle(const Triangulation &triang,
                  map<int, cell_cluster> &cluster_set,
                  const vector<int> &sorted_cluster_index_vector,
                  const int output_seg_count, const char *file_prefix) {
  // we do it in two phases - first output all the tunnels
  // and all the voids as they are more important.
  // next output all the pockets.

  // tunnel and void output.
  // Note: after pruning all small components and voids
  // there should not be any void. But they do appear
  // as the sampling is not good everywhere.
  ofstream fout_seg;
  char *file_suffix = (char *)"_tunnel.off";

  int tun_void_cnt = 0;
  int i = -1;
  for (; tun_void_cnt < output_seg_count;) {
    i++;
    if (i >= (int)sorted_cluster_index_vector.size()) {
      cerr << endl
           << "The number of tunnel/voids are less than " << output_seg_count
           << endl;
      break;
    }
    if (tun_void_cnt >= 100) {
      cerr << "More than 100 tunnel/voids will not be output." << endl;
      break;
    }
    int cl_id = sorted_cluster_index_vector[i];
    if (cluster_set[cl_id].mouth_cnt < 1) {
      cerr << "Cluster " << cl_id << " is a void." << endl;
      file_suffix = (char *)"_void.off";
    } else if (cluster_set[cl_id].mouth_cnt > 1) {
      cerr << "Cluster " << cl_id << " is a tunnel with "
           << cluster_set[cl_id].mouth_cnt << " mouths." << endl;
      file_suffix = (char *)"_tunnel.off";
    } else {
      ; // cerr << "Cluster " << cl_id << " is a pocket." << endl;
      continue;
    }

    char op_fname[100];
    char extn[10];
    extn[0] = '_';
    extn[1] = '0' + tun_void_cnt / 10;
    extn[2] = '0' + tun_void_cnt % 10;
    extn[3] = '\0';
    strcpy(op_fname, file_prefix);
    strcat(op_fname, extn);
    strcat(op_fname, file_suffix);
    cerr << "file : " << op_fname << endl;

    tun_void_cnt++;

    fout_seg.open(op_fname);
    if (!fout_seg) {
      cerr << "Error in opening output file " << endl;
      exit(1);
    }
    // write the ith biggest cluster.

    // do facet count.
    int facet_count = 0;
    for (FFI fit = triang.finite_facets_begin();
         fit != triang.finite_facets_end(); fit++) {
      Cell_handle c[2] = {(*fit).first,
                          (*fit).first->neighbor((*fit).second)};
      if (cluster_set[c[0]->id].find() == cluster_set[c[1]->id].find())
        continue;
      if (cluster_set[c[0]->id].find() != cl_id &&
          cluster_set[c[1]->id].find() != cl_id)
        continue;
      facet_count++;
    }
    // write header.
    // fout_seg << "# " << cl_id << endl;
    fout_seg << "OFF" << endl;
    fout_seg << triang.number_of_vertices() << " " << facet_count << " 0"
             << endl;
    // write the vertices.
    for (FVI vit = triang.finite_vertices_begin();
         vit != triang.finite_vertices_end(); vit++)
      fout_seg << vit->point() << endl;

    // write the facets.
    for (FFI fit = triang.finite_facets_begin();
         fit != triang.finite_facets_end(); fit++) {
      Cell_handle c[2] = {(*fit).first,
                          (*fit).first->neighbor((*fit).second)};
      int id[2] = {c[0]->index(c[1]), c[1]->index(c[0])};

      if (cluster_set[c[0]->id].find() == cluster_set[c[1]->id].find())
        continue;
      if (cluster_set[c[0]->id].find() != cl_id &&
          cluster_set[c[1]->id].find() != cl_id)
        continue;
      // check if it is a pocket/tunnel/void.
      CGAL_assertion(cluster_set[cl_id].mouth_cnt !=
                     1); // it is not a pocket.
      double r, g, b;
      if (cluster_set[cl_id].mouth_cnt > 1) // tunnel
      {
        if (cluster_set[c[0]->id].outside != cluster_set[c[1]->id].outside) {
          r = 1;
          g = 1;
          b = 0; // yellow
        } else {
          r = 1;
          g = 0;
          b = 0; // mouth - red
        }
      } else {
        CGAL_assertion(cluster_set[cl_id].mouth_cnt == 0); // void
        if (c[0]->cocone_flag(id[0])) {
          r = 1;
          g = 0;
          b = 1; // purple
        } else {
          r = 0;
          g = 0;
          b = 1; // blue
        }
      }
      fout_seg << "3\t";
      for (int j = 1; j <= 3; j++)
        fout_seg << (*fit).first->vertex(((*fit).second + j) % 4)->id << " ";
      fout_seg << r << " " << g << " " << b << " 1" << endl;
    }
    fout_seg.close();
  }
  cerr << endl;

  // writing pockets.
  file_suffix = (char *)"_pocket.off";
  int pocket_cnt = 0;
  i = -1;
  for (; pocket_cnt < output_seg_count;) {
    i++;
    if (i >= (int)sorted_cluster_index_vector.size()) {
      cerr << endl << "No more pockets." << endl;
      break;
    }
    if (pocket_cnt >= 100) {
      cerr << "More than 100 pockets will not be output." << endl;
      break;
    }
    int cl_id = sorted_cluster_index_vector[i];
    if (cluster_set[cl_id].mouth_cnt < 1)
      continue;
    else if (cluster_set[cl_id].mouth_cnt > 1)
      continue;
    else
      cerr << "Cluster " << cl_id << " is a pocket." << endl;

    char op_fname[100];
    char extn[10];
    extn[0] = '_';
    extn[1] = '0' + pocket_cnt / 10;
    extn[2] = '0' + pocket_cnt % 10;
    extn[3] = '\0';
    strcpy(op_fname, file_prefix);
    strcat(op_fname, extn);
    strcat(op_fname, file_suffix);
    cerr << "file : " << op_fname << endl;

    pocket_cnt++;

    fout_seg.open(op_fname);
    if (!fout_seg) {
      cerr << "Error in opening output file " << endl;
      exit(1);
    }
    // write the ith biggest cluster.

    // do facet count.
    int facet_count = 0;
    for (FFI fit = triang.finite_facets_begin();
         fit != triang.finite_facets_end(); fit++) {
      Cell_handle c[2] = {(*fit).first,
                          (*fit).first->neighbor((*fit).second)};
      if (cluster_set[c[0]->id].find() == cluster_set[c[1]->id].find())
        continue;
      if (cluster_set[c[0]->id].find() != cl_id &&
          cluster_set[c[1]->id].find() != cl_id)
        continue;
      facet_count++;
    }
    // write header.
    // fout_seg << "# " << cl_id << endl;
    fout_seg << "OFF" << endl;
    fout_seg << triang.number_of_vertices() << " " << facet_count << " 0"
             << endl;
    // write the vertices.
    for (FVI vit = triang.finite_vertices_begin();
         vit != triang.finite_vertices_end(); vit++)
      fout_seg << vit->point() << endl;

    // write the facets.
    for (FFI fit = triang.finite_facets_begin();
         fit != triang.finite_facets_end(); fit++) {
      Cell_handle c[2] = {(*fit).first,
                          (*fit).first->neighbor((*fit).second)};
      if (cluster_set[c[0]->id].find() == cluster_set[c[1]->id].find())
        continue;
      if (cluster_set[c[0]->id].find() != cl_id &&
          cluster_set[c[1]->id].find() != cl_id)
        continue;
      // check if it is a pocket/tunnel/void.
      double r, g, b;
      CGAL_assertion(cluster_set[cl_id].mouth_cnt ==
                     1); // it is not a pocket.

      if (cluster_set[c[0]->id].outside != cluster_set[c[1]->id].outside) {
        r = 0;
        g = 1;
        b = 0; // surface - green
      } else {
        r = 1;
        g = 0;
        b = 1; // mouth - purple
      }
      fout_seg << "3\t";
      for (int j = 1; j <= 3; j++)
        fout_seg << (*fit).first->vertex(((*fit).second + j) % 4)->id << " ";
      fout_seg << r << " " << g << " " << b << " 1" << endl;
    }
    fout_seg.close();
  }
}

void convert_pocket_tunnel_to_rawc_geometry(
    CVCGEOM_NAMESPACE::cvcgeom_t **PTV, const Triangulation &triang,
    map<int, cell_cluster> &cluster_set,
    const vector<int> &sorted_cluster_index_vector, const int num_pockets,
    const int num_tunnels) {
  // Create a single rawc geometry from the handles.
  vector<int> facet_ids;
  vector<float> vertex_colors;
  vertex_colors.resize(3 * triang.number_of_vertices(), 1);
  int facet_count = 0;

  // Load the geometry of the tunnels and voids.
  int tun_void_cnt = 0;
  int i = -1;
  for (; tun_void_cnt < num_tunnels;) {
    i++;
    if (i >= (int)sorted_cluster_index_vector.size()) {
      cerr << endl
           << "The number of tunnel/voids are less than " << num_pockets
           << endl;
      break;
    }
    if (tun_void_cnt >= 100) {
      cerr << "More than 100 tunnel/voids will not be output." << endl;
      break;
    }
    int cl_id = sorted_cluster_index_vector[i];
    if (cluster_set[cl_id].mouth_cnt < 1)
      cerr << "Cluster " << cl_id << " is a void." << endl;
    else if (cluster_set[cl_id].mouth_cnt > 1)
      cerr << "Cluster " << cl_id << " is a tunnel with "
           << cluster_set[cl_id].mouth_cnt << " mouths." << endl;
    else {
      ; // cerr << "Cluster " << cl_id << " is a pocket." << endl;
      continue;
    }
    tun_void_cnt++;

    for (FFI fit = triang.finite_facets_begin();
         fit != triang.finite_facets_end(); fit++) {
      Cell_handle c[2] = {(*fit).first,
                          (*fit).first->neighbor((*fit).second)};
      int id[2] = {c[0]->index(c[1]), c[1]->index(c[0])};
      if (cluster_set[c[0]->id].find() == cluster_set[c[1]->id].find())
        continue;
      if (cluster_set[c[0]->id].find() != cl_id &&
          cluster_set[c[1]->id].find() != cl_id)
        continue;

      // this facet is part of this tunnel/void.
      // point the face away from the cell which is in this cluster.
      Cell_handle temp_c = c[0];
      int temp_id = c[0]->index(c[1]);
      if (cluster_set[c[0]->id].find() != cl_id) {
        temp_c = c[1];
        temp_id = c[1]->index(c[0]);
      }
      int vid[3] = {temp_c->vertex((temp_id + 1) % 4)->id,
                    temp_c->vertex((temp_id + 2) % 4)->id,
                    temp_c->vertex((temp_id + 3) % 4)->id};
      Tetrahedron tet(temp_c->vertex((temp_id + 1) % 4)->point(),
                      temp_c->vertex((temp_id + 2) % 4)->point(),
                      temp_c->vertex((temp_id + 3) % 4)->point(),
                      temp_c->vertex(temp_id)->point());
      if (CGAL::to_double(tet.volume()) > 0) {
        facet_ids.push_back(vid[0]);
        facet_ids.push_back(vid[2]);
        facet_ids.push_back(vid[1]);
        facet_count++;
      } else {
        facet_ids.push_back(vid[0]);
        facet_ids.push_back(vid[1]);
        facet_ids.push_back(vid[2]);
        facet_count++;
      }

      // color.
      CGAL_assertion(cluster_set[cl_id].mouth_cnt !=
                     1); // it is not a pocket.
      float r, g, b;
      if (cluster_set[cl_id].mouth_cnt > 1) // tunnel
      {
        if (cluster_set[c[0]->id].outside != cluster_set[c[1]->id].outside) {
          r = 1;
          g = 1;
          b = 0; // yellow
        } else {
          r = 1;
          g = 0;
          b = 0; // mouth - red
        }
      } else {
        CGAL_assertion(cluster_set[cl_id].mouth_cnt == 0); // void
        if (c[0]->cocone_flag(id[0])) {
          r = 1;
          g = 0;
          b = 1; // purple
        } else {
          r = 0;
          g = 0;
          b = 1; // blue
        }
      }

      // set the color in vertex_colors.
      for (int j = 0; j < 3; j++) {
        if (vertex_colors[3 * vid[j]] == 1 &&
            vertex_colors[3 * vid[j] + 1] == 0 &&
            vertex_colors[3 * vid[j] + 2] == 0)
          continue;
        vertex_colors[3 * vid[j]] = r;
        vertex_colors[3 * vid[j] + 1] = g;
        vertex_colors[3 * vid[j] + 2] = b;
      }
    }
  }

  // Now create the geometry for the pockets.

  int pocket_cnt = 0;
  i = -1;
  for (; pocket_cnt < num_pockets;) {
    i++;
    if (i >= (int)sorted_cluster_index_vector.size()) {
      cerr << endl << "No more pockets." << endl;
      break;
    }
    if (pocket_cnt >= 100) {
      cerr << "More than 100 pockets will not be output." << endl;
      break;
    }
    int cl_id = sorted_cluster_index_vector[i];
    if (cluster_set[cl_id].mouth_cnt < 1)
      continue;
    else if (cluster_set[cl_id].mouth_cnt > 1)
      continue;
    else
      cerr << "Cluster " << cl_id << " is a pocket." << endl;
    pocket_cnt++;

    for (FFI fit = triang.finite_facets_begin();
         fit != triang.finite_facets_end(); fit++) {
      Cell_handle c[2] = {(*fit).first,
                          (*fit).first->neighbor((*fit).second)};
      int id[2] = {c[0]->index(c[1]), c[1]->index(c[0])};
      if (cluster_set[c[0]->id].find() == cluster_set[c[1]->id].find())
        continue;
      if (cluster_set[c[0]->id].find() != cl_id &&
          cluster_set[c[1]->id].find() != cl_id)
        continue;
      Cell_handle temp_c = c[0];
      int temp_id = c[0]->index(c[1]);
      if (cluster_set[c[0]->id].find() != cl_id) {
        temp_c = c[1];
        temp_id = c[1]->index(c[0]);
      }
      int vid[3] = {temp_c->vertex((temp_id + 1) % 4)->id,
                    temp_c->vertex((temp_id + 2) % 4)->id,
                    temp_c->vertex((temp_id + 3) % 4)->id};
      Tetrahedron tet(temp_c->vertex((temp_id + 1) % 4)->point(),
                      temp_c->vertex((temp_id + 2) % 4)->point(),
                      temp_c->vertex((temp_id + 3) % 4)->point(),
                      temp_c->vertex(temp_id)->point());
      if (CGAL::to_double(tet.volume()) > 0) {
        facet_ids.push_back(vid[0]);
        facet_ids.push_back(vid[2]);
        facet_ids.push_back(vid[1]);
        facet_count++;
      } else {
        facet_ids.push_back(vid[0]);
        facet_ids.push_back(vid[1]);
        facet_ids.push_back(vid[2]);
        facet_count++;
      }

      // color.
      CGAL_assertion(cluster_set[cl_id].mouth_cnt == 1); // it is a pocket.
      float r, g, b;
      if (cluster_set[c[0]->id].outside != cluster_set[c[1]->id].outside) {
        r = 0;
        g = 1;
        b = 0; // surface - green
      } else {
        r = 1;
        g = 0;
        b = 1; // mouth - purple
      }

      // set the color in vertex_colors.
      for (int j = 0; j < 3; j++) {
        if (vertex_colors[3 * vid[j]] == 1 &&
            vertex_colors[3 * vid[j] + 1] == 0 &&
            vertex_colors[3 * vid[j] + 2] == 1)
          continue;

        vertex_colors[3 * vid[j]] = r;
        vertex_colors[3 * vid[j] + 1] = g;
        vertex_colors[3 * vid[j] + 2] = b;
      }
    }
    cerr << "Done creating pocket." << endl;
  }

  // Now we have the vertex_ids, facet_ids and vertex_colors all set.
  // Load these in Geometry.
  (*PTV) = new CVCGEOM_NAMESPACE::cvcgeom_t();

  CVCGEOM_NAMESPACE::cvcgeom_t::point_t newVertex;
  CVCGEOM_NAMESPACE::cvcgeom_t::triangle_t newTri;
  CVCGEOM_NAMESPACE::cvcgeom_t::color_t newColor;

  for (FVI vit = triang.finite_vertices_begin();
       vit != triang.finite_vertices_end(); vit++) {

    newVertex[0] = CGAL::to_double(vit->point().x());
    newVertex[1] = CGAL::to_double(vit->point().y());
    newVertex[2] = CGAL::to_double(vit->point().z());

    newColor[0] = vertex_colors[3 * vit->id];
    newColor[1] = vertex_colors[3 * vit->id + 1];
    newColor[2] = vertex_colors[3 * vit->id + 2];

    (*PTV)->points().push_back(newVertex);
    (*PTV)->colors().push_back(newColor);
  }

  for (i = 0; i < facet_count; i++) {
    newTri[0] = facet_ids[3 * i];
    newTri[1] = facet_ids[3 * i + 1];
    newTri[2] = facet_ids[3 * i + 2];
    (*PTV)->triangles().push_back(newTri);
  }

#if 0
   ofstream fout;
   fout.open("debug_output/temp_PTV.coff");

   fout << "COFF" << endl;
   fout << (*PTV)->m_NumTriVerts << " " << (*PTV)->m_NumTris << " 0" << endl;

   for(i = 0; i < (*PTV)->m_NumTriVerts; i ++)
   {
      fout << (*PTV)->m_TriVerts[3*i] << " "
           << (*PTV)->m_TriVerts[3*i+1] << " "
           << (*PTV)->m_TriVerts[3*i+2] << " "
           << (*PTV)->m_TriVertColors[3*i] << " "
           << (*PTV)->m_TriVertColors[3*i+1] << " "
           << (*PTV)->m_TriVertColors[3*i+2] << " 1" << endl;
   }
   for(i = 0; i < (*PTV)->m_NumTris; i ++)
   {
      fout << "3\t";
      fout << (*PTV)->m_Tris[3*i] << " "
           << (*PTV)->m_Tris[3*i+1] << " "
           << (*PTV)->m_Tris[3*i+2] << endl;
   }
#endif
}

}; // namespace PocketTunnel
