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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#include <Skeletonization/op.h>

namespace Skeletonization {

extern vector<double> bounding_box;

void draw_ray(const Ray &ray, const double &r, const double &g,
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

  fout << "OFF";
  fout << "  " << triang.number_of_vertices(); // The number of points
  fout << " " << num_facets;                   // The number of facets
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

//-----------------
// write_iobdy
//-----------------
// Write out the boundary between inside and outside tetrehedra.
//-----------------------------------------------------------------------
void write_iobdy(const Triangulation &triang, const char *file_prefix) {

  char filename[100];
  strcat(strcpy(filename, file_prefix), ".io");

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
    if (ch->outside != ch->neighbor(id)->outside)
      num_facets++;
  }

  // The header of the output file

  fout << "OFF";
  fout << "  " << triang.number_of_vertices(); // The number of points
  fout << " " << num_facets;                   // The number of facets
  fout << " 0" << endl;

  // Write the vertices.
  for (FVI vit = triang.finite_vertices_begin();
       vit != triang.finite_vertices_end(); ++vit)
    fout << vit->point() << endl;

  for (FFI fit = triang.finite_facets_begin();
       fit != triang.finite_facets_end(); ++fit) {
    Cell_handle ch = (*fit).first;
    int id = (*fit).second;

    if (ch->outside == ch->neighbor(id)->outside)
      continue;

    fout << " 3\t";
    for (int i = 1; i <= 3; i++)
      fout << " " << ch->vertex((id + i) % 4)->id;
    fout << "\t " << "1 1 1 1 \n";
    ;
  }

  fout.close();
}

//-----------------------------------------------------------------------
// write_axis
// -------------
// Writes the Voronoi facets choosed by dual Denaulay edges of cocone and
// ratio
// ---------------------------------------------------------------------------

void write_axis(const Triangulation &triang, const char *file_prefix) {
  char filename[100];
  strcat(strcpy(filename, file_prefix), ".ax");

  ofstream fout;
  fout.open(filename);

  if (!fout) {
    cerr << "Can not open " << filename << " for writing. " << endl;
    exit(1);
  }

  fout << "{LIST" << endl;
  for (FEI eit = triang.finite_edges_begin();
       eit != triang.finite_edges_end(); eit++) {
    Cell_handle c = (*eit).first;
    int uid = (*eit).second, vid = (*eit).third;
    if (!c->VF_on_medax(uid, vid))
      continue;

    if (is_inf_VF(triang, c, uid, vid))
      continue;

    double r = 1, g = 1, b = 1, a = 1;
    bool out = true, in = true;
    bool outside_bb_flag = false;

    Facet_circulator fcirc = triang.incident_facets((*eit));
    Facet_circulator begin = fcirc;
    do {
      if ((*fcirc).first->outside)
        in = false;
      else
        out = false;
      if (is_outside_bounding_box((*fcirc).first->voronoi(), bounding_box))
        outside_bb_flag = true;
      fcirc++;
    } while (fcirc != begin);
    if (outside_bb_flag)
      continue;

    // A medial axis VF can not cross the surface.
    // So it's either fully in ( ! out) or fully out ( ! in ).
    CGAL_assertion(!in || !out);

    if (!in) // outside medax
    {
      r = 0;
      g = 1;
      b = 1;
      a = 0.2;
      draw_VF(triang, (*eit), r, g, b, a, fout);
    } else // inside medax
    {
      r = 1;
      g = 0.7;
      b = 0.7;
      a = 1;
      draw_VF(triang, (*eit), r, g, b, a, fout);
    }
  }
  fout << "}" << endl;
}

void write_u2(const Triangulation &triang, const Graph &graph,
              const char *file_prefix) {
  char filename[255];
  strcat(strcpy(filename, file_prefix), ".u2");

  ofstream fout;
  fout.open(filename);

  if (!fout) {
    cerr << "Can not open " << filename << " for writing. " << endl;
    exit(1);
  }

  fout << "{LIST" << endl;
  for (int i = 0; i < (int)graph.get_ne(); i++) {
    int u = graph.edge_list[i].get_endpoint(0),
        v = graph.edge_list[i].get_endpoint(1);

    if (is_outside_bounding_box(graph.vert_list[u].point(), bounding_box) ||
        is_outside_bounding_box(graph.vert_list[v].point(), bounding_box))
      continue;

    CGAL_assertion(graph.vert_list[u].out() == graph.vert_list[v].out());
    if (graph.vert_list[u].out() && graph.vert_list[v].out())
      continue;

    fout << "{OFF" << endl;
    fout << "2 1 0" << endl;
    fout << graph.vert_list[u].point() << endl;
    fout << graph.vert_list[v].point() << endl;
    fout << "2\t0 1 ";

    vector<int> C1 = graph.vert_list[u].cluster_membership;
    vector<int> C2 = graph.vert_list[v].cluster_membership;
    if (is_there_any_common_element(C1, C2))
      fout << "0 1 1 0.1" << endl; // transparent.
    else
      fout << "0 1 1 1" << endl; // opaque.
    fout << "}" << endl;
  }
  fout << "}" << endl;
  fout.close();
}

void write_u1(const Triangulation &triang, const char *file_prefix) {
  char op_filename[100];
  strcat(strcpy(op_filename, file_prefix), ".u1");
  ofstream fout;
  fout.open(op_filename);
  fout << "{LIST" << endl;

  for (FEI eit = triang.finite_edges_begin();
       eit != triang.finite_edges_end(); eit++) {
    if (!eit->first->VF_on_um_i1(eit->second, eit->third))
      continue;

    // color the VF differently depending on of it is IN, OUT or MIXED.
    double r = 1, g = 1, b = 1, a = 1;
    bool out = false, in = false;
    bool outside_bb_flag = false;

    Facet_circulator fcirc = triang.incident_facets((*eit));
    Facet_circulator begin = fcirc;
    do {
      if ((*fcirc).first->outside)
        out = true;
      else
        in = true;
      if (is_outside_bounding_box((*fcirc).first->voronoi(), bounding_box))
        outside_bb_flag = true;
      fcirc++;
    } while (fcirc != begin);

    if (outside_bb_flag)
      continue;
    CGAL_assertion((!in || !out) && (in || out));

    if (in) {
      r = 0;
      g = 1;
      b = 0;
      a = 1;
    } else {
#ifndef __OUTSIDE__
      continue;
#endif
      r = 1;
      g = 0;
      b = 0;
      a = 0.2;
    }
    draw_VF(triang, (*eit), r, g, b, a, fout);
  }
  fout << "}" << endl;
}

void write_skel(const Skel &skel, const char *file_prefix) {
  char op_filename[100];
  strcat(strcpy(op_filename, file_prefix), ".skel");
  ofstream fout;
  fout.open(op_filename);
  fout << "{LIST" << endl;
  // srand48(0);
  srand(0);

  // fout << "# PLANAR " << endl << endl;
  for (int i = 0; i < (int)skel.sorted_pl_id.size(); i++) {
    int cl_id = skel.sorted_pl_id[i];
    if (!skel.is_big_pl[cl_id]) {
      // draw the star.
      // fout << "# --- " << endl;
      Point c = skel.pl_C[cl_id];
      for (int j = 0; j < (int)skel.active_bdy[cl_id].size(); j++)
        draw_segment(Segment(c, skel.active_bdy[cl_id][j]), 0, 0, 1, 0.5,
                     fout);
      // fout << "# --- " << endl;
    } else {
      // fout << "# --- " << endl;
      // double r = drand48(), g = drand48(), b = drand48(), a = 1;
      double r = double(rand()) / double(RAND_MAX),
             g = double(rand()) / double(RAND_MAX),
             b = double(rand()) / double(RAND_MAX), a = 1;
      for (int j = 0; j < (int)skel.pl[cl_id].size(); j++) {
        double scale_color = skel.pl[cl_id][j].width / skel.max_pgn_width;
        // draw_poly(skel.pl[cl_id][j].ordered_v_list,
        // r*scale_color, g*scale_color, b*scale_color, a, fout);
        draw_poly(skel.pl[cl_id][j].ordered_v_list, r, g, b, a, fout);
      }
      // fout << "# --- " << endl;
    }
  }

  // fout << "# LINEAR " << endl << endl;
  // write the linear part.
  for (int i = 0; i < (int)skel.L.size(); i++) {
    Polyline l = skel.L[i];
    bool is_far = false;
    for (int j = 0; j < (int)l.ordered_v_list.size(); j++)
      if (is_outside_bounding_box(l.ordered_v_list[j], bounding_box))
        is_far = true;
    if (is_far)
      continue;
    // draw the polyline with a random color.
    // double r = drand48(), g = drand48(), b = drand48(), a = 1;
    double r = double(rand()) / double(RAND_MAX),
           g = double(rand()) / double(RAND_MAX),
           b = double(rand()) / double(RAND_MAX), a = 1;
    r = 0;
    g = 1;
    b = 0;
    if (skel.L_invalid[i])
      continue; // a = 0.1;

    for (int j = 0; j < (int)l.ordered_v_list.size() - 1; j++)
      draw_segment(Segment(l.ordered_v_list[j], l.ordered_v_list[j + 1]), r,
                   g, b, a, fout);
  }

  for (int i = 0; i < (int)skel.L.size(); i++) {
    if (skel.L_invalid[i])
      continue;
    for (int j = 0; j < (int)skel.L[i].cell_list.size(); j++) {
      Cell_handle c = skel.L[i].cell_list[j];
      if (c->big_pl()) {
        continue;
      }
      if (sqrt(c->cell_radius()) >= 2.5 + 0.5 ||
          sqrt(c->cell_radius()) <= 2.5 - 0.5)
        continue;
      fout << "appearance {linewidth 4}" << endl;
      fout << "{OFF" << endl;
      fout << "1 1 0" << endl;
      fout << skel.L[i].ordered_v_list[j] << endl;
      if (is_maxima(c))
        fout << "1\t0 1 0 0 1" << endl;
      else
        fout << "1\t0 0 1 0 1" << endl;
      fout << "}" << endl;
    }
  }
  fout << "}" << endl;
}

void write_pl_cyl(const Triangulation &triang, const char *file_prefix) {
  char filename[100];
  strcat(strcpy(filename, file_prefix), ".pl_cyl");

  ofstream fout;
  fout.open(filename);

  fout << triang.number_of_vertices() << " " << triang.number_of_vertices()
       << " 0" << endl;

  for (FVI vit = triang.finite_vertices_begin();
       vit != triang.finite_vertices_end(); ++vit)
    fout << vit->point() << endl;
  for (FVI vit = triang.finite_vertices_begin();
       vit != triang.finite_vertices_end(); ++vit)
    if (vit->is_pl())
      fout << "1\t" << vit->id << " 0 1 0 1" << endl;
    else if (vit->is_cyl())
      fout << "1\t" << vit->id << " 0 0 1 0.5" << endl;
    else
      fout << "1\t" << vit->id << " 1 0 0 0.2" << endl;
}

void write_pl_cyl_surf(const Triangulation &triang, const char *file_prefix) {
  char filename[100];
  strcat(strcpy(filename, file_prefix), ".pl_cyl_surf");

  ofstream fout;
  fout.open(filename);

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

  fout << "COFF";
  fout << "  " << triang.number_of_vertices(); // The number of points
  fout << " " << num_facets;                   // The number of facets
  fout << " 0" << endl;

  // Write the vertices.
  for (FVI vit = triang.finite_vertices_begin();
       vit != triang.finite_vertices_end(); ++vit) {
    fout << vit->point() << " ";
    if (vit->is_pl())
      fout << "\t" << " 0.33 0.88 0.7 1" << endl;
    else if (vit->is_cyl())
      fout << "\t" << " 0.941 0.788 0.176 1" << endl;
    else
      fout << "\t" << " 1 1 1 0.1" << endl;
    // fout << "\t" << " 0.33 0.88 0.7 1" << endl; // color it as planar.
  }

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

void write_helices(const Triangulation &triang,
                   map<int, cell_cluster> &cluster_set,
                   const vector<int> &sorted_cluster_index_vector,
                   const int &helix_cnt, const char *file_prefix) {
  // While writing the segments create the color-plate.
  // For each segment there will be one color.
  vector<float> r_vector;
  vector<float> g_vector;
  vector<float> b_vector;

  // make a color plate
  for (unsigned int i = 0; i < sorted_cluster_index_vector.size(); i++) {
    // srand48(sorted_cluster_index_vector[i]);
    srand(sorted_cluster_index_vector[i]);
    // float r = drand48(), g = drand48(), b = drand48();
    double r = double(rand()) / double(RAND_MAX),
           g = double(rand()) / double(RAND_MAX),
           b = double(rand()) / double(RAND_MAX);
    r_vector.push_back(r);
    g_vector.push_back(g);
    b_vector.push_back(b);
  }

  for (int i = 0; i < helix_cnt; i++) {
    if (i >= 100) {
      cerr << "more than 100 segments will not be output." << endl;
      break;
    }
    if (i > (int)sorted_cluster_index_vector.size()) {
      cerr << endl
           << "The number of segments are less than " << helix_cnt << endl;
      break;
    }

    char op_fname[100];
    char extn[10];
    extn[0] = '_';
    extn[1] = '0' + i / 10;
    extn[2] = '0' + i % 10;
    extn[3] = '\0';
    strcpy(op_fname, file_prefix);
    strcat(op_fname, extn);
    strcat(op_fname, ".alpha");
    cerr << "file : " << op_fname << endl;

    ofstream fout_seg;
    fout_seg.open(op_fname);
    if (!fout_seg) {
      cerr << "Error in opening output file " << endl;
      exit(1);
    }

    // do facet count.
    int facet_count = 0;
    for (FFI fit = triang.finite_facets_begin();
         fit != triang.finite_facets_end(); fit++) {
      Cell_handle ch = (*fit).first;
      int id = (*fit).second;
      if (!ch->in_helix_cluster() && !ch->neighbor(id)->in_helix_cluster())
        continue;
      if (cluster_set[ch->id].find() ==
          cluster_set[ch->neighbor(id)->id].find())
        continue;
      if (cluster_set[ch->id].find() == sorted_cluster_index_vector[i] ||
          cluster_set[ch->neighbor(id)->id].find() ==
              sorted_cluster_index_vector[i])
        facet_count++;
    }

    fout_seg << "OFF\n\n";
    fout_seg << triang.number_of_vertices() << " " << facet_count << " 0\n\n";
    // write the vertices.
    for (FVI vit = triang.finite_vertices_begin();
         vit != triang.finite_vertices_end(); vit++)
      fout_seg << vit->point() << endl;
    // write the facets.
    for (FFI fit = triang.finite_facets_begin();
         fit != triang.finite_facets_end(); fit++) {
      Cell_handle ch = (*fit).first;
      int id = (*fit).second;
      if (!ch->in_helix_cluster() && !ch->neighbor(id)->in_helix_cluster())
        continue;
      if (cluster_set[ch->id].find() ==
          cluster_set[ch->neighbor(id)->id].find())
        continue;
      if (cluster_set[ch->id].find() != sorted_cluster_index_vector[i] &&
          cluster_set[ch->neighbor(id)->id].find() !=
              sorted_cluster_index_vector[i])
        continue;
      fout_seg << "3\t";
      for (int j = 1; j <= 3; j++)
        fout_seg << (*fit).first->vertex(((*fit).second + j) % 4)->id << " ";
      fout_seg << r_vector[i] << " " << g_vector[i] << " " << b_vector[i]
               << " 0.5" << endl;
    }
    fout_seg.close();
  }
}

} // namespace Skeletonization
