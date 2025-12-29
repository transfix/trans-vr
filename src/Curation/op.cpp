/*
  Copyright 2008 The University of Texas at Austin

        Authors: Samrat Goswami <samrat@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Curation.

  Curation is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  Curation is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#include <Curation/op.h>

namespace Curation {

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

void write_smax(const Triangulation &triang,
                map<int, cell_cluster> &cluster_set,
                const vector<int> &sorted_cluster_index_vector,
                const int output_seg_count, const char *file_prefix) {
  ofstream fout_seg;
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

  char *file_suffix = (char *)"_seg.off";
  for (int i = 0; i < output_seg_count; i++) {
    if (i >= (int)sorted_cluster_index_vector.size()) {
      cerr << endl
           << "The number of segments are less than " << output_seg_count
           << endl;
      break;
    }
    if (i >= 100) {
      cerr << "more than 100 segments will not be output." << endl;
      break;
    }
    int cl_id = sorted_cluster_index_vector[i];
    char op_fname[100];
    char extn[10];
    extn[0] = '_';
    extn[1] = '0' + i / 10;
    extn[2] = '0' + i % 10;
    extn[3] = '\0';
    strcpy(op_fname, file_prefix);
    strcat(op_fname, extn);
    strcat(op_fname, file_suffix);
    cerr << "file : " << op_fname << endl;

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
      double r = r_vector[i], g = g_vector[i], b = b_vector[i];
      fout_seg << "3\t";
      for (int j = 1; j <= 3; j++)
        fout_seg << (*fit).first->vertex(((*fit).second + j) % 4)->id << " ";
      fout_seg << r << " " << g << " " << b << " 1" << endl;
    }
    fout_seg.close();

    /*
    // convert to wrl.
    fout_seg.open("temp");
    fout_seg << triang.number_of_vertices() << " "
    << facet_count << " 0" << endl;
    // write the vertices.
    for(FVI vit = triang.finite_vertices_begin();
    vit != triang.finite_vertices_end(); vit ++)
    fout_seg << vit->point() << endl;

    // write the facets.
    for(FFI fit = triang.finite_facets_begin();
    fit != triang.finite_facets_end(); fit ++)
    {
    Cell_handle c[2] = {(*fit).first, (*fit).first->neighbor((*fit).second)};
    int id[2] = {c[0]->index(c[1]), c[1]->index(c[0])};

    if(cluster_set[c[0]->id].find() ==
    cluster_set[c[1]->id].find()) continue;
    if(cluster_set[c[0]->id].find() != cl_id &&
    cluster_set[c[1]->id].find() != cl_id )
    continue;
    // check if it is a pocket/tunnel/void.
    double r = r_vector[i],
    g = g_vector[i],
    b = b_vector[i];
    fout_seg << "3\t";
    for(int j = 1; j <=3; j ++)
    fout_seg << (*fit).first->vertex(((*fit).second + j)%4)->id << " ";
    fout_seg << r << " " << g << " " << b << " 1" << endl;
    }

    fout_seg.close();

    char off2wrl_command[200] = "./off_to_wrlV2 ";
    strcat( off2wrl_command, "temp" );
    char wrl_fname[100];
    strcpy(wrl_fname, file_prefix);
    strcat(wrl_fname, extn);
    strcat(wrl_fname, ".seg.wrl");
    strcat( off2wrl_command, " ");
    strcat( off2wrl_command, wrl_fname );
    system(off2wrl_command);
    */
  }
  cerr << endl;
  fout_seg.close();
}

void write_mesh(const Mesh &mesh, const char *op_prefix) {
  char filename[100];
  strcpy(filename, op_prefix);
  strcat(filename, ".mesh");
  ofstream fout;
  fout.open(filename);
  fout << "{LIST" << endl;
  fout << "{OFF" << endl;
  fout << mesh.get_nv() << " " << mesh.get_nf() << " 0" << endl;
  for (int i = 0; i < mesh.get_nv(); i++) {
    double x = mesh.vert_list[i].point()[0];
    double y = mesh.vert_list[i].point()[1];
    double z = mesh.vert_list[i].point()[2];
    fout << x << " " << y << " " << z << endl;
  }
  for (int i = 0; i < mesh.get_nf(); i++) {
    int v1 = mesh.face_list[i].get_corner(0);
    int v2 = mesh.face_list[i].get_corner(1);
    int v3 = mesh.face_list[i].get_corner(2);
    fout << "3\t" << v1 << " " << v2 << " " << v3 << " ";
    if (mesh.vert_list[v1].v_nm() || mesh.vert_list[v2].v_nm() ||
        mesh.vert_list[v3].v_nm())
      fout << " 1 0 0 1" << endl;
    else if (mesh.vert_list[v1].e_nm() || mesh.vert_list[v2].e_nm() ||
             mesh.vert_list[v3].e_nm())
      fout << " 0 1 0 1" << endl;
    else
      fout << " 1 1 1 0.3" << endl;
  }
  fout << "}" << endl;
  fout << "}" << endl;
}

CVCGEOM_NAMESPACE::cvcgeom_t
write_smax_to_geom(const Triangulation &triang,
                   map<int, cell_cluster> &cluster_set,
                   const vector<int> &sorted_cluster_index_vector) {
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

  //   std::vector<CVCGEOM_NAMESPACE::cvcgeom_t > results;

  /*   for(int i = 0; i < output_seg_count; i ++)
       {
         if(i >= (int)sorted_cluster_index_vector.size())
           {
             cerr << endl << "The number of segments are less than " <<
     output_seg_count << endl; break;
           }
         if(i >= 100)
           {
             cerr << "more than 100 segments will not be output." << endl;
             break;
           }
         int cl_id = sorted_cluster_index_vector[i];
   */
  /*	// do facet count.
          int facet_count = 0;
          for(FFI fit = triang.finite_facets_begin();
              fit != triang.finite_facets_end(); fit ++)
            {
              Cell_handle c[2] = {(*fit).first,
     (*fit).first->neighbor((*fit).second)}; if(cluster_set[c[0]->id].find()
     == cluster_set[c[1]->id].find()) continue;
              if(cluster_set[c[0]->id].find() != cl_id &&
                 cluster_set[c[1]->id].find() != cl_id )
                continue;
              facet_count ++;
            }

           cout<< "facet number is: " << facet_count << endl; */
  /*
         boost::shared_ptr<Geometry> geom(new Geometry());
         geom->AllocateTris(triang.number_of_vertices(),
                            facet_count);  */

  CVCGEOM_NAMESPACE::cvcgeom_t result;
  // write the vertices.

  CVCGEOM_NAMESPACE::cvcgeom_t::point_t newVertex;
  CVCGEOM_NAMESPACE::cvcgeom_t::triangle_t newTri;

  //	int cnt = 0;
  for (FVI vit = triang.finite_vertices_begin();
       vit != triang.finite_vertices_end(); vit++) {
    newVertex[0] = vit->point().x();
    newVertex[1] = vit->point().y();
    newVertex[2] = vit->point().z();

    result.points().push_back(newVertex);
  }

  // write the facets.
  //	cnt = 0;
  for (FFI fit = triang.finite_facets_begin();
       fit != triang.finite_facets_end(); fit++) {
    Cell_handle c[2] = {(*fit).first, (*fit).first->neighbor((*fit).second)};
    int id[2] = {c[0]->index(c[1]), c[1]->index(c[0])};
    if (c[0]->outside == c[1]->outside)
      continue;
    Vertex_handle vh[3] = {c[0]->vertex((id[0] + 1) % 4),
                           c[0]->vertex((id[0] + 2) % 4),
                           c[0]->vertex((id[0] + 3) % 4)};
    if (!c[0]->outside) {
      if (CGAL::is_negative(Tetrahedron(vh[0]->point(), vh[1]->point(),
                                        vh[2]->point(),
                                        c[0]->vertex(id[0])->point())
                                .volume()))
      //          fout << "3\t" << vh[0]->id << " " << vh[1]->id << " " <<
      //          vh[2]->id << " ";
      // fout << "3\t" << vh[0]->id << " " << vh[1]->id << " " << vh[2]->id <<
      // " ";
      {
        newTri[0] = vh[0]->id;
        newTri[1] = vh[1]->id;
        newTri[2] = vh[2]->id;
        result.triangles().push_back(newTri);
      } else {
        newTri[0] = vh[1]->id;
        newTri[1] = vh[0]->id;
        newTri[2] = vh[2]->id;
        result.triangles().push_back(newTri);

        //	fout << "3\t" << vh[1]->id << " " << vh[0]->id << " " <<
        //vh[2]->id << " ";
      }
    } else {
      if (CGAL::is_negative(
              Tetrahedron(vh[0]->point(), vh[1]->point(), vh[2]->point(),
                          c[1]->vertex(id[1])->point())
                  .volume())) { //    fout << "3\t" << vh[0]->id << " " <<
                                //    vh[1]->id << " " << vh[2]->id << " ";
        newTri[0] = vh[0]->id;
        newTri[1] = vh[1]->id;
        newTri[2] = vh[2]->id;
        result.triangles().push_back(newTri);
      } else {
        //            fout << "3\t" << vh[1]->id << " " << vh[0]->id << " " <<
        //            vh[2]->id << " ";
        newTri[0] = vh[1]->id;
        newTri[1] = vh[0]->id;
        newTri[2] = vh[2]->id;
        result.triangles().push_back(newTri);
      }
    }
    /*	    if(cluster_set[c[0]->id].find() ==
                   cluster_set[c[1]->id].find()) continue;
                if(cluster_set[c[0]->id].find() != cl_id &&
                   cluster_set[c[1]->id].find() != cl_id )
                  continue;
                // check if it is a pocket/tunnel/void.
                double r = r_vector[i],
                  g = g_vector[i],
                  b = b_vector[i];
                //fout_seg << "3\t";
                for(int j = 1; j <=3; j ++)
                  geom->m_Tris[3*cnt+(j-1)] =
       (*fit).first->vertex(((*fit).second + j)%4)->id;
                  //fout_seg << (*fit).first->vertex(((*fit).second +
       j)%4)->id << " ";
                //fout_seg << r << " " << g << " " << b << " 1" << endl;

                cnt++;*/
  }

  return result;
}

} // namespace Curation
