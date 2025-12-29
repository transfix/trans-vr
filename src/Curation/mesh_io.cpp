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

#include <Curation/mesh_io.h>
#include <cstdio>

namespace Curation {
// ------------------------
// read_mesh
// ------------------------
void read_mesh(Mesh &mesh, const char *ip_filename, bool read_color_opacity) {
  FILE *fp = fopen(ip_filename, "r");
  char header[10];
  fscanf(fp, "%s", header);
  if (strcmp(header, "OFF") != 0) {
    cerr << "Error : Missing header OFF" << endl;
    exit(1);
  }
  int nv = -1, nf = -1, skip_int = -1;
  fscanf(fp, "%d %d %d", &nv, &nf, &skip_int);
  mesh.set_nv(nv);
  mesh.set_nf(nf);
  for (int i = 0; i < nv; i++) {
    float x, y, z;
    fscanf(fp, "%f %f %f", &x, &y, &z);
    mesh.vert_list.push_back(MVertex(Point(x, y, z)));
    mesh.vert_list[mesh.vert_list.size() - 1].id = i;
  }
  int ne = 0;
  for (int i = 0; i < nf; i++) {
    int np = -1, v1 = -1, v2 = -1, v3 = -1;
    float r, g, b, a;
    if (read_color_opacity)
      fscanf(fp, "%d %d %d %d %f %f %f %f", &np, &v1, &v2, &v3, &r, &g, &b,
             &a);
    else {
      fscanf(fp, "%d %d %d %d", &np, &v1, &v2, &v3);
      r = 1;
      g = 1;
      b = 1;
      a = 1;
    }
    if (np != 3) {
      cerr << endl << "Please check the file format. " << endl;
      cerr << "This program accepts only OFF format with triangles." << endl;
      exit(1);
    }
    mesh.face_list.push_back(MFace(v1, v2, v3));
    mesh.face_list[mesh.face_list.size() - 1].id = i;
    // set the isolated tag false at these vertices.
    mesh.vert_list[v1].set_iso(false);
    mesh.vert_list[v2].set_iso(false);
    mesh.vert_list[v3].set_iso(false);
    // book keeping for incident vertices for a vertex.
    if (!mesh.vert_list[v1].is_inc_vert(v2)) {
      mesh.vert_list[v1].add_inc_vert(v2);
      assert(!mesh.vert_list[v2].is_inc_vert(v1));
      mesh.vert_list[v2].add_inc_vert(v1);

      // here create an edge between v1 and v2.
      // add the face to the edge
      MEdge me = MEdge(v1, v2);
      me.add_inc_face(i);
      mesh.edge_list.push_back(me);
      ne++;

      // insert the id of the incident edge in the vertices
      mesh.vert_list[v1].add_inc_edge(mesh.edge_list.size() - 1);
      mesh.vert_list[v2].add_inc_edge(mesh.edge_list.size() - 1);
    } else {
      // the edge is already there.
      // find the edge id using the vertex indices.
      MVertex mv = mesh.vert_list[v1];
      int eid = -1;
      assert(mv.get_eid(v2, eid));
      assert(eid != -1);
      // add the face to the edge.
      if (mesh.edge_list[eid].num_inc_face <= 1) {
        assert(mesh.edge_list[eid].num_inc_face == 1);
        mesh.edge_list[eid].add_inc_face(i);
      } else {
        mesh.edge_list[eid].set_non_manifold_edge(true);
        mesh.edge_list[eid].num_inc_face++;
      }
    }

    if (!mesh.vert_list[v2].is_inc_vert(v3)) {
      mesh.vert_list[v2].add_inc_vert(v3);
      assert(!mesh.vert_list[v3].is_inc_vert(v2));
      mesh.vert_list[v3].add_inc_vert(v2);

      // here create an edge between v2 and v3.
      // add the face to the edge
      MEdge me = MEdge(v2, v3);
      me.add_inc_face(i);
      mesh.edge_list.push_back(me);
      ne++;

      // insert the id of the incident edge in the vertices
      mesh.vert_list[v2].add_inc_edge(mesh.edge_list.size() - 1);
      mesh.vert_list[v3].add_inc_edge(mesh.edge_list.size() - 1);
    } else {
      // the edge is already there.
      // find the edge id using the vertex indices.
      MVertex mv = mesh.vert_list[v2];
      int eid = -1;
      assert(mv.get_eid(v3, eid));
      assert(eid != -1);
      // add the face to the edge.
      if (mesh.edge_list[eid].num_inc_face <= 1) {
        assert(mesh.edge_list[eid].num_inc_face == 1);
        mesh.edge_list[eid].add_inc_face(i);
      } else {
        mesh.edge_list[eid].set_non_manifold_edge(true);
        mesh.edge_list[eid].num_inc_face++;
      }
    }

    if (!mesh.vert_list[v3].is_inc_vert(v1)) {
      mesh.vert_list[v3].add_inc_vert(v1);
      assert(!mesh.vert_list[v1].is_inc_vert(v3));
      mesh.vert_list[v1].add_inc_vert(v3);

      // here create an edge between v3 and v1.
      // add the face to the edge
      MEdge me = MEdge(v3, v1);
      me.add_inc_face(i);
      mesh.edge_list.push_back(me);
      ne++;

      // insert the id of the incident edge in the vertices
      mesh.vert_list[v3].add_inc_edge(mesh.edge_list.size() - 1);
      mesh.vert_list[v1].add_inc_edge(mesh.edge_list.size() - 1);
    } else {
      // the edge is already there.
      // find the edge id using the vertex indices.
      MVertex mv = mesh.vert_list[v3];
      int eid = -1;
      assert(mv.get_eid(v1, eid));
      assert(eid != -1);
      // add the face to the edge.
      if (mesh.edge_list[eid].num_inc_face <= 1) {
        assert(mesh.edge_list[eid].num_inc_face == 1);
        mesh.edge_list[eid].add_inc_face(i);
      } else {
        mesh.edge_list[eid].set_non_manifold_edge(true);
        mesh.edge_list[eid].num_inc_face++;
      }
    }
  }
  mesh.set_ne(ne);
  // add the edges in the face so that
  // if v1 is the ith corner of a face
  // then v2<->v3 is the ith edge.
  for (int i = 0; i < nf; i++) {
    for (int j = 0; j < 3; j++) {
      int u = mesh.face_list[i].get_corner((j + 1) % 3);
      int w = mesh.face_list[i].get_corner((j + 2) % 3);
      // find the edge id connecting u amd w.
      int eid = -1;
      assert(mesh.vert_list[u].get_eid(w, eid));
      assert(eid != -1);
      // this edge should be the jth edge of the face.
      mesh.face_list[i].set_edge(j, eid);
    }
  }
  fclose(fp);
}

/*  void
  read_mesh_from_geom(Mesh &mesh,const boost::shared_ptr<Geometry>& geom)
  {

    printf("Mesh %d %d\n", geom->m_NumTriVerts,geom->m_NumTris);

    mesh.set_nv(geom->m_NumTriVerts);
    mesh.set_nf(geom->m_NumTris);
    int nv = geom->m_NumTriVerts;
    int nf = geom->m_NumTris;
    for(int i = 0; i < nv; i ++)
      {
        mesh.vert_list.push_back(MVertex(Point(geom->m_TriVerts[3*i+0],
                                               geom->m_TriVerts[3*i+1],
                                               geom->m_TriVerts[3*i+2])));
        mesh.vert_list[mesh.vert_list.size() - 1].id = i;
      }
    int ne = 0;
    for(int i = 0; i < nf; i ++)
      {
        int v1 = -1, v2 = -1, v3 = -1;
        v1 = geom->m_Tris[3*i+0];
        v2 = geom->m_Tris[3*i+1];
        v3 = geom->m_Tris[3*i+2];

        mesh.face_list.push_back(MFace(v1,v2,v3));
        mesh.face_list[mesh.face_list.size() - 1].id = i;
        // set the isolated tag false at these vertices.
        mesh.vert_list[v1].set_iso(false);
        mesh.vert_list[v2].set_iso(false);
        mesh.vert_list[v3].set_iso(false);

        // book keeping for incident vertices for a vertex.
        if(!mesh.vert_list[v1].is_inc_vert(v2))
          {
            mesh.vert_list[v1].add_inc_vert(v2);
            assert(!mesh.vert_list[v2].is_inc_vert(v1));
            mesh.vert_list[v2].add_inc_vert(v1);


            // here create an edge between v1 and v2.
            // add the face to the edge
            MEdge me = MEdge(v1,v2);
            me.add_inc_face(i);
            mesh.edge_list.push_back(me);
            ne ++;

            // insert the id of the incident edge in the vertices
            mesh.vert_list[v1].add_inc_edge(mesh.edge_list.size() - 1);
            mesh.vert_list[v2].add_inc_edge(mesh.edge_list.size() - 1);
          }
        else
          {
            // the edge is already there.
            // find the edge id using the vertex indices.
            MVertex mv = mesh.vert_list[v1];
            int eid = -1;
            assert(mv.get_eid(v2, eid));
            assert(eid != -1);
            // add the face to the edge.
            if(mesh.edge_list[eid].num_inc_face <= 1)
              {
                assert(mesh.edge_list[eid].num_inc_face == 1);
                mesh.edge_list[eid].add_inc_face(i);
              }
            else
              {
                mesh.edge_list[eid].set_non_manifold_edge(true);
                mesh.edge_list[eid].num_inc_face ++;
              }
          }

        if(!mesh.vert_list[v2].is_inc_vert(v3))
          {
            mesh.vert_list[v2].add_inc_vert(v3);
            assert(!mesh.vert_list[v3].is_inc_vert(v2));
            mesh.vert_list[v3].add_inc_vert(v2);

            // here create an edge between v2 and v3.
            // add the face to the edge
            MEdge me = MEdge(v2,v3);
            me.add_inc_face(i);
            mesh.edge_list.push_back(me);
            ne ++;

            // insert the id of the incident edge in the vertices
            mesh.vert_list[v2].add_inc_edge(mesh.edge_list.size() - 1);
            mesh.vert_list[v3].add_inc_edge(mesh.edge_list.size() - 1);
          }
        else
          {
            // the edge is already there.
            // find the edge id using the vertex indices.
            MVertex mv = mesh.vert_list[v2];
            int eid = -1;
            assert(mv.get_eid(v3, eid));
            assert(eid != -1);
            // add the face to the edge.
            if(mesh.edge_list[eid].num_inc_face <= 1)
              {
                assert(mesh.edge_list[eid].num_inc_face == 1);
                mesh.edge_list[eid].add_inc_face(i);
              }
            else
              {
                mesh.edge_list[eid].set_non_manifold_edge(true);
                mesh.edge_list[eid].num_inc_face ++;
              }
          }

        if(!mesh.vert_list[v3].is_inc_vert(v1))
          {
            mesh.vert_list[v3].add_inc_vert(v1);
            assert(!mesh.vert_list[v1].is_inc_vert(v3));
            mesh.vert_list[v1].add_inc_vert(v3);

            // here create an edge between v3 and v1.
            // add the face to the edge
            MEdge me = MEdge(v3,v1);
            me.add_inc_face(i);
            mesh.edge_list.push_back(me);
            ne ++;

            // insert the id of the incident edge in the vertices
            mesh.vert_list[v3].add_inc_edge(mesh.edge_list.size() - 1);
            mesh.vert_list[v1].add_inc_edge(mesh.edge_list.size() - 1);
          }
        else
          {
            // the edge is already there.
            // find the edge id using the vertex indices.
            MVertex mv = mesh.vert_list[v3];
            int eid = -1;
            assert(mv.get_eid(v1, eid));
            assert(eid != -1);
            // add the face to the edge.
            if(mesh.edge_list[eid].num_inc_face <= 1)
              {
                assert(mesh.edge_list[eid].num_inc_face == 1);
                mesh.edge_list[eid].add_inc_face(i);
              }
            else
              {
                mesh.edge_list[eid].set_non_manifold_edge(true);
                mesh.edge_list[eid].num_inc_face ++;
              }
          }
      }

    mesh.set_ne(ne);
    // add the edges in the face so that
    // if v1 is the ith corner of a face
    // then v2<->v3 is the ith edge.
    for(int i = 0; i < nf; i ++)
      {
        for(int j = 0; j < 3; j ++)
          {
            int u = mesh.face_list[i].get_corner((j+1)%3);
            int w = mesh.face_list[i].get_corner((j+2)%3);
            // find the edge id connecting u amd w.
            int eid = -1;
            assert(mesh.vert_list[u].get_eid(w, eid));
            assert(eid != -1);
            // this edge should be the jth edge of the face.
            mesh.face_list[i].set_edge(j, eid);
          }
      }

  } */

void read_mesh_from_geom(Mesh &mesh,
                         const CVCGEOM_NAMESPACE::cvcgeom_t &geom) {

  printf("Mesh %d %d\n", geom.points().size(), geom.triangles().size());

  int nv = geom.points().size();
  int nf = geom.triangles().size();

  mesh.set_nv(nv);
  mesh.set_nf(nf);

  for (int i = 0; i < nv; i++) {
    mesh.vert_list.push_back(MVertex(Point(
        geom.points()[i][0], geom.points()[i][1], geom.points()[i][2])));
    mesh.vert_list[mesh.vert_list.size() - 1].id = i;
  }
  int ne = 0;
  for (int i = 0; i < nf; i++) {
    int v1 = -1, v2 = -1, v3 = -1;
    v1 = geom.triangles()[i][0];
    v2 = geom.triangles()[i][1];
    v3 = geom.triangles()[i][2];

    mesh.face_list.push_back(MFace(v1, v2, v3));
    mesh.face_list[mesh.face_list.size() - 1].id = i;
    // set the isolated tag false at these vertices.
    mesh.vert_list[v1].set_iso(false);
    mesh.vert_list[v2].set_iso(false);
    mesh.vert_list[v3].set_iso(false);

    // book keeping for incident vertices for a vertex.
    if (!mesh.vert_list[v1].is_inc_vert(v2)) {
      mesh.vert_list[v1].add_inc_vert(v2);
      assert(!mesh.vert_list[v2].is_inc_vert(v1));
      mesh.vert_list[v2].add_inc_vert(v1);

      // here create an edge between v1 and v2.
      // add the face to the edge
      MEdge me = MEdge(v1, v2);
      me.add_inc_face(i);
      mesh.edge_list.push_back(me);
      ne++;

      // insert the id of the incident edge in the vertices
      mesh.vert_list[v1].add_inc_edge(mesh.edge_list.size() - 1);
      mesh.vert_list[v2].add_inc_edge(mesh.edge_list.size() - 1);
    } else {
      // the edge is already there.
      // find the edge id using the vertex indices.
      MVertex mv = mesh.vert_list[v1];
      int eid = -1;
      assert(mv.get_eid(v2, eid));
      assert(eid != -1);
      // add the face to the edge.
      if (mesh.edge_list[eid].num_inc_face <= 1) {
        assert(mesh.edge_list[eid].num_inc_face == 1);
        mesh.edge_list[eid].add_inc_face(i);
      } else {
        mesh.edge_list[eid].set_non_manifold_edge(true);
        mesh.edge_list[eid].num_inc_face++;
      }
    }

    if (!mesh.vert_list[v2].is_inc_vert(v3)) {
      mesh.vert_list[v2].add_inc_vert(v3);
      assert(!mesh.vert_list[v3].is_inc_vert(v2));
      mesh.vert_list[v3].add_inc_vert(v2);

      // here create an edge between v2 and v3.
      // add the face to the edge
      MEdge me = MEdge(v2, v3);
      me.add_inc_face(i);
      mesh.edge_list.push_back(me);
      ne++;

      // insert the id of the incident edge in the vertices
      mesh.vert_list[v2].add_inc_edge(mesh.edge_list.size() - 1);
      mesh.vert_list[v3].add_inc_edge(mesh.edge_list.size() - 1);
    } else {
      // the edge is already there.
      // find the edge id using the vertex indices.
      MVertex mv = mesh.vert_list[v2];
      int eid = -1;
      assert(mv.get_eid(v3, eid));
      assert(eid != -1);
      // add the face to the edge.
      if (mesh.edge_list[eid].num_inc_face <= 1) {
        assert(mesh.edge_list[eid].num_inc_face == 1);
        mesh.edge_list[eid].add_inc_face(i);
      } else {
        mesh.edge_list[eid].set_non_manifold_edge(true);
        mesh.edge_list[eid].num_inc_face++;
      }
    }

    if (!mesh.vert_list[v3].is_inc_vert(v1)) {
      mesh.vert_list[v3].add_inc_vert(v1);
      assert(!mesh.vert_list[v1].is_inc_vert(v3));
      mesh.vert_list[v1].add_inc_vert(v3);

      // here create an edge between v3 and v1.
      // add the face to the edge
      MEdge me = MEdge(v3, v1);
      me.add_inc_face(i);
      mesh.edge_list.push_back(me);
      ne++;

      // insert the id of the incident edge in the vertices
      mesh.vert_list[v3].add_inc_edge(mesh.edge_list.size() - 1);
      mesh.vert_list[v1].add_inc_edge(mesh.edge_list.size() - 1);
    } else {
      // the edge is already there.
      // find the edge id using the vertex indices.
      MVertex mv = mesh.vert_list[v3];
      int eid = -1;
      assert(mv.get_eid(v1, eid));
      assert(eid != -1);
      // add the face to the edge.
      if (mesh.edge_list[eid].num_inc_face <= 1) {
        assert(mesh.edge_list[eid].num_inc_face == 1);
        mesh.edge_list[eid].add_inc_face(i);
      } else {
        mesh.edge_list[eid].set_non_manifold_edge(true);
        mesh.edge_list[eid].num_inc_face++;
      }
    }
  }

  mesh.set_ne(ne);
  // add the edges in the face so that
  // if v1 is the ith corner of a face
  // then v2<->v3 is the ith edge.
  for (int i = 0; i < nf; i++) {
    for (int j = 0; j < 3; j++) {
      int u = mesh.face_list[i].get_corner((j + 1) % 3);
      int w = mesh.face_list[i].get_corner((j + 2) % 3);
      // find the edge id connecting u amd w.
      int eid = -1;
      assert(mesh.vert_list[u].get_eid(w, eid));
      assert(eid != -1);
      // this edge should be the jth edge of the face.
      mesh.face_list[i].set_edge(j, eid);
    }
  }
}

} // namespace Curation
