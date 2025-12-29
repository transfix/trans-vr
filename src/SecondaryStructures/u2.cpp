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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#include <SecondaryStructures/u2.h>

using namespace SecondaryStructures;

extern vector<double> bounding_box;

void output_status_on_screen(const COMPUTATION_STATUS &status) {
  if (status == COSPH) {
    cerr << "<Cosph>";
  } else if (status == NOT_FOUND) {
    cerr << "<! Found>";
  } else if (status == INTERSECT_PROBLEM) {
    cerr << "<Bad Int>";
  } else if (status == SAME_P) {
    cerr << "<Comp Err>";
  } else if (status == ERROR1) {
    cerr << "<Algo Prob>";
  }
  return;
}

bool has_cospherical_pair_of_tetrahedra_dual_to_any_VE(
    const Triangulation &triang, const vector<Cell_handle> &VF) {
  for (int i = 0; i < (int)VF.size(); i++) {
    int id[2] = {VF[i]->index(VF[(i + 1) % ((int)VF.size())]),
                 VF[(i + 1) % ((int)VF.size())]->index(VF[i])};
    if (is_cospherical_pair(triang, Facet(VF[i], id[0]))) {
      CGAL_assertion(is_cospherical_pair(
          triang, Facet(VF[(i + 1) % ((int)VF.size())], id[1])));
      return true;
    }
  }
  return false;
}

bool find_target_VE(const vector<Cell_handle> &VF,
                    const vector<Cell_handle> &forbidden_VE, const Point &d,
                    const Point &p, Cell_handle &c0, Cell_handle &c1,
                    const char *degen_op_filename) {
  bool found_target_VE = false;
  for (int i = 0; i < (int)VF.size(); i++) {
    if ((int)forbidden_VE.size() == 2)
      if ((VF[i]->id == forbidden_VE[0]->id &&
           VF[(i + 1) % ((int)VF.size())]->id == forbidden_VE[1]->id) ||
          (VF[i]->id == forbidden_VE[1]->id &&
           VF[(i + 1) % ((int)VF.size())]->id == forbidden_VE[0]->id)) {
        continue;
      }
    if ((int)forbidden_VE.size() == 1)
      if ((VF[i]->id == forbidden_VE[0]->id ||
           VF[(i + 1) % ((int)VF.size())]->id == forbidden_VE[0]->id)) {
        continue;
      }
    if (does_intersect_ray3_seg3_in_plane(
            Ray_3(d, p),
            Segment(VF[i]->voronoi(),
                    VF[(i + 1) % ((int)VF.size())]->voronoi()))) {
      c0 = VF[i];
      c1 = VF[(i + 1) % ((int)VF.size())];
      found_target_VE = true;
    }
  }
  if (!found_target_VE) {
    ofstream fout;
    fout.open(degen_op_filename);
    fout << "{LIST" << endl;
    draw_segment(Segment(d, p), 1, 0, 0, 1, fout);
    vector<Point> poly;
    for (int i = 0; i < (int)VF.size(); i++) {
      poly.push_back(VF[i]->voronoi());
    }
    draw_poly(poly, 1, 1, 0, 1, fout);
    fout << "}" << endl;
  }
  return found_target_VE;
}

Point get_intersection_point(const Point &d, const Point &p,
                             const Cell_handle &c0, const Cell_handle &c1,
                             bool &is_correct_intersection,
                             const vector<Cell_handle> &VF,
                             const char *degen_op_filename) {
  Point new_p =
      intersect_ray3_seg3(Ray_3(d, p), Segment(c0->voronoi(), c1->voronoi()),
                          is_correct_intersection);
  if (!is_correct_intersection) {
    ofstream fout;
    fout.open(degen_op_filename);
    fout << "{LIST" << endl;
    draw_segment(Segment(d, p), 1, 0, 0, 1, fout);
    draw_segment(Segment(c0->voronoi(), c1->voronoi()), 0, 1, 0, 1, fout);
    vector<Point> poly;
    for (int i = 0; i < (int)VF.size(); i++) {
      poly.push_back(VF[i]->voronoi());
    }
    draw_poly(poly, 1, 1, 0, 1, fout);
    fout << "}" << endl;
    return CGAL::ORIGIN;
  }
  return new_p;
}

// Edge e holds the starting VV and VF it will flow into.
COMPUTATION_STATUS compute_nonVE_integral_curve(const Triangulation &triang,
                                                Point &d, Edge e,
                                                vector<Cell_handle> &chain,
                                                Cell_handle &dest,
                                                char *prefix) {
  char no_target_VE_degen_filename[255];
  strcat(strcpy(no_target_VE_degen_filename, prefix), ".no_target_VE.degen");
  char incorrect_intersection_degen_filename[255];
  strcat(strcpy(incorrect_intersection_degen_filename, prefix),
         ".incorrect_intersection.degen");
  char same_p_degen_filename[255];
  strcat(strcpy(same_p_degen_filename, prefix), ".same_p.degen");
  Cell_handle c = e.first;
  Point p = c->voronoi();
  // helps excluding the VE from intersection calculation
  // on which p lies.
  vector<Cell_handle> forbidden_VE;
  // Handle initial forbidden edges
  forbidden_VE.clear();
  forbidden_VE.push_back(c);
  while (true) {
    // create VF dual to e. store the cells.
    vector<Cell_handle> VF;
    Facet_circulator fcirc = triang.incident_facets(e);
    Facet_circulator begin = fcirc;
    do {
      if (triang.is_infinite((*fcirc).first)) {
        return INFINITE1;
      }
      VF.push_back((*fcirc).first);
      if ((*fcirc).first->dirty()) {
        cerr << " <dirty VF> ";
      }
      fcirc++;
    } while (fcirc != begin);
    // avoid the obstacles one by one.
    // 1. cospherical - we are not handling it now.
    if (has_cospherical_pair_of_tetrahedra_dual_to_any_VE(triang, VF)) {
      return COSPH;
    }
    // 2. If fails while trying to find a target VE in the VF.
    Cell_handle c0, c1;
    if (!find_target_VE(VF, forbidden_VE, d, p, c0, c1,
                        no_target_VE_degen_filename)) {
      return NOT_FOUND;
    }
    // 3. If numerical instability hinders the calculation, i.e., when
    // incorrect intersection.
    bool is_correct_intersection = true;
    Point new_p =
        get_intersection_point(d, p, c0, c1, is_correct_intersection, VF,
                               incorrect_intersection_degen_filename);
    if (!is_correct_intersection) {
      return INTERSECT_PROBLEM;
    }
    // 4. If the intersection point is same as the starting point in the VF -
    // potential Inf loop.
    if (p == new_p) {
      ofstream fout;
      fout.open(same_p_degen_filename);
      fout << "{LIST" << endl;
      draw_segment(Segment(d, p), 1, 0, 0, 1, fout);
      draw_segment(Segment(c0->voronoi(), c1->voronoi()), 0, 1, 0, 1, fout);
      draw_VF(triang, e, 1, 1, 0, 1, fout);
      fout << "}" << endl;
      return SAME_P;
    }
    // Update the chain and assign new_p to p.
    p = new_p;
    // chain.push_back(p);
    // Hunt for the next segment on this path.
    // The VE on which p (or new_p) lies can be transversal or
    // non-transversal. If it is non-transversal, we can uniquely identify the
    // next VV the flow will hit via the current VE. We collect the dual DT
    // and return with SUCCESS.
    int id = c0->index(c1);
    if (!is_transversal_flow(Facet(c0, id))) {
      // p will flow to c0/c1 depending on the half-plane it is in.
      if (CGAL::orientation(c0->vertex((id + 1) % 4)->point(),
                            c0->vertex((id + 2) % 4)->point(),
                            c0->vertex((id + 3) % 4)->point(),
                            c0->vertex(id)->point()) ==
          CGAL::orientation(c0->vertex((id + 1) % 4)->point(),
                            c0->vertex((id + 2) % 4)->point(),
                            c0->vertex((id + 3) % 4)->point(), p)) {
        dest = c0;
      } else {
        dest = c1;
      }
      return SUCCESS;
    }
    // If it is transversal, we need to recur the same routine for the
    // Acceptor VF. update d. d = driver of VE(c0->voronoi(), c1->voronoi())
    int uid, vid, wid = -1;
    CGAL_assertion(find_acceptor(c0, id, uid, vid, wid));
    CGAL_assertion(uid + vid + wid + id == 6);
    d = CGAL::midpoint(c0->vertex(uid)->point(), c0->vertex(vid)->point());
    // update e.
    // e = dual(acceptor(VE(c0->voronoi(), c1->voronoi())))
    e = Edge(c0, uid, vid);
    // update the forbidden_VE(s).
    forbidden_VE.clear();
    forbidden_VE.push_back(c0);
    forbidden_VE.push_back(c1);
  }
  return ERROR1;
}

void sort_i2set_by_circumradius(const vector<Facet> &i2set,
                                vector<Facet> &sorted_i2set) {
  vector<bool> b;
  b.resize((int)i2set.size(), false);
  for (int i = 0; i < (int)i2set.size(); i++) {
    if (i % 200 == 0) {
      cerr << " ";
    }
    int ind = -1;
    double min = HUGE;
    for (int j = 0; j < (int)i2set.size(); j++) {
      if (b[j]) {
        continue;
      }
      if (circumradius(i2set[j]) < min) {
        min = circumradius(i2set[j]);
        ind = j;
      }
    }
    CGAL_assertion(ind != -1);
    CGAL_assertion(min != HUGE);
    b[ind] = true;
    sorted_i2set.push_back(Facet(i2set[ind].first, i2set[ind].second));
  }
}

// Identify the index-2 saddles in the voronoi diagrams and compute their
// unstable manifolds.
pair<vector<vector<Cell_handle>>, vector<Facet>>
compute_u2(Triangulation &triang, char *prefix) {
  vector<vector<Cell_handle>> chains;
  vector<COMPUTATION_STATUS> status_property;
  vector<Facet> start_of_chains;
  cerr << "\ti2 saddles ";
  // set of index-2 saddle points.
  vector<Facet> i2set;
  for (FFI fit = triang.finite_facets_begin();
       fit != triang.finite_facets_end(); fit++) {
    Cell_handle c[2];
    int id[2];
    c[0] = (*fit).first;
    id[0] = (*fit).second;
    c[1] = c[0]->neighbor(id[0]);
    id[1] = c[1]->index(c[0]);
    c[0]->visited = false;
    c[1]->visited = false;
    if (triang.is_infinite(c[0]) || triang.is_infinite(c[1])) {
      continue;
    }
    if (!c[0]->VV_on_medax() || !c[1]->VV_on_medax()) {
      continue;
    }
    // only in:
    // if(c[0]->outside || c[1]->outside)
    // continue;
    // only out:
    // if( ! c[0]->outside || ! c[1]->outside)
    // continue;
    if (!is_i2_saddle((*fit))) {
      continue;
    }
    // put this facet whose circumcenter is an index-2 saddle into a stack.
    i2set.push_back(Facet(c[0], id[0]));
  }
  cerr << "collected.";
  // holds i2saddles (small to big).
  vector<Facet> sorted_i2set;
  // sort_i2set_by_circumradius(i2set, sorted_i2set);
  sorted_i2set = i2set;
  cerr << "sorted. ";
  // number of chain components is twice the number of i2 saddle points.
  chains.resize(2 * (int)i2set.size());
  status_property.resize(2 * (int)i2set.size(), SUCCESS);
  start_of_chains.resize(2 * (int)i2set.size());
  int i2cnt = -1;
  int progress = 0;
  while (!sorted_i2set.empty()) {
    i2cnt++;
    Facet i2f = sorted_i2set.back();
    sorted_i2set.pop_back();
    if (++progress % 1000 == 0) {
      progress = 0;
      cerr << ".";
    }
    vector<Cell_handle> subchains[2];
    subchains[0].push_back(i2f.first);
    subchains[1].push_back(i2f.first->neighbor(i2f.second));
    i2f.first->visited = true;
    i2f.first->neighbor(i2f.second)->visited = true;
    start_of_chains[2 * i2cnt] = i2f;
    start_of_chains[2 * i2cnt + 1] = i2f;
    // check if the VE holding the i2saddle point crosses the surface.
    if (i2f.first->outside != i2f.first->neighbor(i2f.second)->outside) {
      status_property[2 * i2cnt] = status_property[2 * i2cnt + 1] = SURF;
    }
    for (int i = 0; i < 2; i++) {
      bool current_subchain_is_already_traced = false;
      while (!subchains[i].empty()) {
        Cell_handle c = subchains[i].back();
        subchains[i].pop_back();
        CGAL_assertion(c->visited);
        // if the cell is at infinity continue.
        if (triang.is_infinite(c)) {
          continue;
        }
        chains[2 * i2cnt + i].push_back(c);
        if (current_subchain_is_already_traced) {
          continue;
        }
        // a subchain is augmented one-by-one.
        // so, after taking out one vertex, it should be empty.
        CGAL_assertion(subchains[i].empty());
        // reminder: voronoi computation was not correct.
        if (c->dirty()) {
          cerr << "< dirty >";
        }
        int out_cnt = 0;
        int ofid[2] = {-1, -1};
        // detect the outflows.
        for (int j = 0; j < 4; j++)
          if (is_outflow(Facet(c, j))) {
            ofid[out_cnt++] = j;
          }
        CGAL_assertion(out_cnt <= 2);
        // if the flow is transversal, we follow this convention.
        // convention : uid,vid corresponds to the ACCEPTOR DE.
        //              uid,wid and vid,wid correspond to the DONOR DE.
        //           => angle(u,w,v) > 90 degree
        int uid = -1, vid = -1, wid = -1;
        bool flow_through_VF = false;
        if (out_cnt == 0) {
          CGAL_assertion(is_maxima(c));
        } else if (out_cnt == 1) {
          // special attention - degenerate case: cospherical.
          if (is_cospherical_pair(triang, Facet(c, ofid[0]))) {
            continue;
          }
          // end special attention.
          CGAL_assertion(ofid[0] != -1 && ofid[1] == -1);
          if (!is_transversal_flow(Facet(c, ofid[0]))) {
            subchains[i].push_back(c->neighbor(ofid[0]));
            if (c->neighbor(ofid[0])->visited) {
              current_subchain_is_already_traced = true;
            }
            c->neighbor(ofid[0])->visited = true;
            if (c->outside != c->neighbor(ofid[0])->outside) {
              status_property[2 * i2cnt + i] = SURF;
            }
          } else // flow through VF.
          {
            CGAL_assertion(find_acceptor(c, ofid[0], uid, vid, wid));
            // set flow_through_VF = true to continue the flow.
            flow_through_VF = true;
          }
        } else if (out_cnt == 2) {
          // special attention
          if (is_cospherical_pair(triang, Facet(c, ofid[0]))) {
            continue;
          }
          if (is_cospherical_pair(triang, Facet(c, ofid[1]))) {
            continue;
          }
          // end special attention
          CGAL_assertion(ofid[0] != -1 && ofid[1] != -1 &&
                         ofid[0] != ofid[1]);
          if (!is_transversal_flow(Facet(c, ofid[0])) &
              !is_transversal_flow(Facet(c, ofid[1]))) // degenerate case.
          {
            cerr << "< degenerate >";
          } else if (!is_transversal_flow(Facet(c, ofid[0]))) {
            // flow will be through VE dual to Facet(c, ofid[0])
            subchains[i].push_back(c->neighbor(ofid[0]));
            if (c->neighbor(ofid[0])->visited) {
              current_subchain_is_already_traced = true;
            }
            c->neighbor(ofid[0])->visited = true;
            if (c->outside != c->neighbor(ofid[0])->outside) {
              status_property[2 * i2cnt + i] = SURF;
            }
          } else if (!is_transversal_flow(Facet(c, ofid[1]))) {
            // flow will be through the edge.
            subchains[i].push_back(c->neighbor(ofid[1]));
            if (c->neighbor(ofid[1])->visited) {
              current_subchain_is_already_traced = true;
            }
            c->neighbor(ofid[1])->visited = true;
            if (c->outside != c->neighbor(ofid[1])->outside) {
              status_property[2 * i2cnt + i] = SURF;
            }
          } else // flow through VF.
          {
            vertex_indices(ofid[0], ofid[1], uid, vid);
            wid = ofid[1]; // arbitrary assignment of wid.
            CGAL_assertion(uid != -1 && vid != -1 && uid != vid);
            // sanity check:
            // If dual(DF1) and dual(DF2) both transversal and
            //  - ACC(dual(DF1)) == ACC(dual(DF2)).
            Point p[4];
            p[0] = c->vertex(uid)->point();
            p[1] = c->vertex(vid)->point();
            p[2] = c->vertex(ofid[0])->point();
            p[3] = c->vertex(ofid[1])->point();
            CGAL_assertion(is_obtuse(p[0], p[1], p[2]) &&
                           is_obtuse(p[0], p[1], p[3]));
            // end - sanity check.
            // set flow_through_VF = true to continue the flow.
            flow_through_VF = true;
          }
        } else {
          CGAL_assertion(false);
        }
        if (flow_through_VF) {
          Point driver = CGAL::midpoint(c->vertex(uid)->point(),
                                        c->vertex(vid)->point());
          if (is_degenerate_VF(triang, c, ofid[0], uid, vid, driver,
                               prefix)) {
            continue;
          }
          Cell_handle dest_c;
          COMPUTATION_STATUS status = compute_nonVE_integral_curve(
              triang, driver, Edge(c, uid, vid), chains[2 * i2cnt + i],
              dest_c, prefix);
          if (status != SUCCESS) {
            status_property[2 * i2cnt + i] = status;
            output_status_on_screen(status);
            continue;
          }
          subchains[i].push_back(dest_c);
          if (dest_c->visited) {
            current_subchain_is_already_traced = true;
          }
          dest_c->visited = true;
          if (c->outside != dest_c->outside) {
            status_property[2 * i2cnt + i] = SURF;
          }
        }
      }
    }
  }
  /*
  // temporary - output the chains.
  cout << "{LIST" << endl;
  cout << "# " << chains.size() << endl;
  for(int i = 0; i < (int)chains.size(); i ++)
  {
     if(status_property[i] != SUCCESS &&
        status_property[i] != SURF )
        continue;
     Point p = circumcenter(start_of_chains[i]);
     if( is_outside_bounding_box(p, bounding_box) ||
         is_outside_bounding_box(chains[i][0]->voronoi(), bounding_box) )
        continue;

     double r,g,b,a;
     if(status_property[i] == SURF)
     {
        r = 1; g = 0; b = 0; a = 0.5;
     }
     else
     {
        if(chains[i][0]->outside)
        {
           r = 0; g = 0; b = 1; a = 0;
        }
        else
        {
           r = 0; g = 1; b = 0; a = 1;
        }
     }


     cout << "# ---------------- " << endl;
     cout << "# " << circumradius(start_of_chains[i]) << endl;
     cout << "{OFF" << endl;
     cout << "2 1 0" << endl;
     cout << circumcenter(start_of_chains[i]) << endl;
     cout << chains[i][0]->voronoi() << endl;
     cout << "2\t0 1 ";
     cout << r << " " << g << " " << b << " " << a << endl;
     cout << "}" << endl;

     for(int j = 0; j < (int)chains[i].size() - 1; j ++)
     {
        Cell_handle c[2] = {chains[i][j], chains[i][j+1]};

        if( triang.is_infinite(c[0]) ||
            triang.is_infinite(c[1]) )
           continue;

        if( is_outside_bounding_box( c[0]->voronoi(), bounding_box) ||
            is_outside_bounding_box( c[1]->voronoi(), bounding_box) )
           continue;

        cout << "{OFF" << endl;
        cout << "2 1 0" << endl;
        cout << c[0]->voronoi() << endl;
        cout << c[1]->voronoi() << endl;
        cout << "2\t0 1 ";
        cout << r << " " << g << " " << b << " " << a << endl;
        cout << "}" << endl;
     }
     cout << "# ---------------- " << endl;
  }
  cout << "}" << endl;
  // end temporary
  */
  vector<vector<Cell_handle>> C; // chains with SUCCESS.
  vector<Facet> S;               // chains with SUCCESS.
  for (int i = 0; i < (int)chains.size(); i++)
    if (status_property[i] == SUCCESS) {
      C.push_back(chains[i]);
      S.push_back(start_of_chains[i]);
    }
  pair<vector<vector<Cell_handle>>, vector<Facet>> u2(C, S);
  return u2;
}
