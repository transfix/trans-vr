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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include <SecondaryStructures/op.h>

using namespace SecondaryStructures;

void draw_Ray_3(const Ray_3& Ray_3, const double& r, const double& g,	const double& b, const double& a, ofstream& fout)
{
	fout << "{OFF" << endl;
	fout << "2 1 0" << endl;
	fout << Ray_3.source() << endl;
	fout << (Ray_3.source() - CGAL::ORIGIN) + Ray_3.to_vector() << endl;
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


void draw_poly(const vector<Point>& poly, const double& r, const double& g, const double& b, const double& a, ofstream& fout)
{
	fout << "{OFF" << endl;
	fout << (int)poly.size() << " 1 0" << endl;
	for (int i = 0; i < (int)poly.size(); i ++)
	{
		fout << poly[i] << endl;
	}
	fout << (int)poly.size() << "\t";
	for (int i = 0; i < (int)poly.size(); i ++)
	{
		fout << i << " ";
	}
	fout << r << " " << g << " " << b << " " << a << endl;
	fout << "}" << endl;
}


void draw_VF(const Triangulation& triang, const Edge& dual_e, const double& r, const double& g, const double& b, const double& a, ofstream& fout)
{
	Facet_circulator fcirc = triang.incident_facets(dual_e);
	Facet_circulator begin = fcirc;
	vector<Point> vvset;
	do
	{
		Cell_handle cc = (*fcirc).first;
		vvset.push_back(cc->voronoi());
		fcirc ++;
	}
	while (fcirc != begin);
	fout << "{OFF" << endl;
	fout << (int)vvset.size() << " 1 0" << endl;
	for (int i = 0; i < (int)vvset.size(); i ++)
	{
		fout << vvset[i] << endl;
	}
	fout << (int)vvset.size() << "\t";
	for (int i = 0; i < (int)vvset.size(); i ++)
	{
		fout << i << " ";
	}
	fout << r << " " << g << " " << b << " " << a << endl;
	fout << "}" << endl;
}

void draw_tetra(const Cell_handle& cell, const double& r, const double& g, const double& b, const double& a, ofstream& fout)
{
	fout << "{OFF" << endl;
	fout << "4 4 0" << endl;
	for (int i = 0; i < 4; i ++)
	{
		fout << cell->vertex(i)->point() << endl;
	}
	for (int i = 0; i < 4; i ++)
		fout << "3\t" << (i+1)%4 << " "
			 << (i+2)%4 << " "
			 << (i+3)%4 << " "
			 << r << " " << g << " " << b << " " << a << endl;
	fout << "}" << endl;
}

// Write out the boundary between inside and outside tetrehedra as surface.
void write_wt(const Triangulation& triang, const char* file_prefix)
{
	char filename[100];
	strcat(strcpy(filename, file_prefix), ".surf");
	ofstream fout;
	fout.open(filename);
	if (! fout)
	{
		cerr << "Can not open " << filename << " for writing. " << endl;
		exit(1);
	}
	// Count number of facets on the surface.
	int num_facets = 0;
	for (FFI fit = triang.finite_facets_begin();
			fit != triang.finite_facets_end(); ++fit)
		if ((*fit).first->cocone_flag((*fit).second))
		{
			num_facets ++;
		}
	fout <<"OFF" << endl;
	fout << triang.number_of_vertices()
		 << " " << num_facets << " 0" << endl;
	// Write the vertices.
	for (FVI vit = triang.finite_vertices_begin();
			vit != triang.finite_vertices_end(); ++vit)
	{
		fout << vit->point() << endl;
	}
	// Write the facets.
	for (FFI fit = triang.finite_facets_begin();
			fit != triang.finite_facets_end(); ++fit)
	{
		Cell_handle c[2] = {(*fit).first, (*fit).first->neighbor((*fit).second)};
		int id[2] = {c[0]->index(c[1]), c[1]->index(c[0])};
		if (! c[0]->cocone_flag(id[0]))
		{
			continue;
		}
		CGAL_assertion(c[0]->bdy[id[0]] && c[1]->bdy[id[1]]);
		CGAL_assertion(c[0]->outside != c[1]->outside);
		Vertex_handle vh[3] = { c[0]->vertex((id[0]+1)%4),
								c[0]->vertex((id[0]+2)%4),
								c[0]->vertex((id[0]+3)%4)
							  };
		if (! c[0]->outside)
			if (CGAL::is_negative(Tetrahedron(vh[0]->point(), vh[1]->point(), vh[2]->point(),
											  c[0]->vertex(id[0])->point()).volume()))
			{
				fout << "3\t" << vh[0]->id << " " << vh[1]->id << " " << vh[2]->id << " ";
			}
			else
			{
				fout << "3\t" << vh[1]->id << " " << vh[0]->id << " " << vh[2]->id << " ";
			}
		else if (CGAL::is_negative(Tetrahedron(vh[0]->point(), vh[1]->point(), vh[2]->point(),
											   c[1]->vertex(id[1])->point()).volume()))
		{
			fout << "3\t" << vh[0]->id << " " << vh[1]->id << " " << vh[2]->id << " ";
		}
		else
		{
			fout << "3\t" << vh[1]->id << " " << vh[0]->id << " " << vh[2]->id << " ";
		}
		fout << "1 1 1 0.3" << endl;
	}
	fout.close();
}
void write_helix_wrl(cvcraw_geometry::cvcgeom_t* geom, char* filename)
{
/*	ofstream fout;
	fout.open(filename);
	if(!fout)
	{
		cerr<<"Cannot open " << filename << " for writing. "<<endl;
		exit(1);
	}
	fout<<"#VRML V2.0 utf8 "<<endl;
	fout<<"Shape {" << endl;
	fout<<"\t appearance Appearance {" << endl;
	fout<<"\t\t material Material {" << endl;
	fout<<"\t\t\t ambientIntensity 0.8" << endl;
	fout<<"\t\t\t diffuseColor 0.99 0.0 0.7"<<endl;
	fout<<"\t\t\t specularColor	0.8 0.8 0.01" <<endl;
	fout<<"\t\t\t emissiveColor	0.0 0.01 0.01" << endl;
	fout<<"\t\t\t shininess	0.5" << endl;
	fout<<"\t\t\t transparency	0"<<endl;
    fout<<"\t\t }" <<endl;
    fout<<"\t }"<<endl;
    fout<<"\t geometry IndexedFaceSet {" <<endl;
	fout<<"\t\t creaseAngle 1.0" <<endl;
    fout<<"\t\t coord Coordinate { " <<endl;
	fout<<"\t\t\t point [" <<endl;
	for(int i = 0; i < geom->m_NumTriVerts; i++)
		fout<<geom->m_TriVerts[3*i+0] <<" " << geom->m_TriVerts[3*i+1] <<" " << geom->m_TriVerts[3*i+2] << endl;
	fout<<"\t\t\t ]"<<endl;
	fout<<"\t\t }" <<endl;
	fout<<"\t\t coordIndex [" << endl;
	for(int i = 0; i < geom->m_NumTris; i++)
		fout<<geom->m_Tris[3*i+0] <<" "<<geom->m_Tris[3*i+1] <<" "<<geom->m_Tris[3*i+2] <<" -1" <<endl;
	fout<<"\t\t ]"<<endl;
	fout<<"\t }" << endl;
	fout<<"}" << endl;
	fout.close(); */

}



void write_sheet_wrl(cvcraw_geometry::cvcgeom_t* geom, char* filename)
{
/*	ofstream fout;
	fout.open(filename);
	if(!fout)
	{
		cerr<<"Cannot open " << filename << " for writing. "<<endl;
		exit(1);
	}
	fout<<"#VRML V2.0 utf8 "<<endl;
	fout<<"Shape {" << endl;
	fout<<"\t appearance Appearance {" << endl;
	fout<<"\t\t material Material {" << endl;
	fout<<"\t\t\t ambientIntensity 0.8" << endl;
	fout<<"\t\t\t diffuseColor 0.99 0.0 0.7"<<endl;
	fout<<"\t\t\t specularColor	0.8 0.8 0.01" <<endl;
	fout<<"\t\t\t emissiveColor	0.0 0.01 0.01" << endl;
	fout<<"\t\t\t shininess	0.5" << endl;
	fout<<"\t\t\t transparency	0"<<endl;
    fout<<"\t\t }" <<endl;
    fout<<"\t }"<<endl; 
    fout<<"\t geometry IndexedFaceSet {" <<endl;
	fout<<"\t\t creaseAngle 1.0" <<endl;
    fout<<"\t\t coord Coordinate { " <<endl;
	fout<<"\t\t\t point [" <<endl;
	for(int i = 0; i < geom->m_NumTriVerts; i++)
		fout<<geom->m_TriVerts[3*i+0] <<" " << geom->m_TriVerts[3*i+1] <<" " << geom->m_TriVerts[3*i+2] << endl;
	fout<<"\t\t\t ]"<<endl;
	fout<<"\t\t }" <<endl;
	fout<<"\t solid FALSE" << endl;
	fout<<"\t colorPerVertex TRUE" << endl;
	fout<<"\t\t color Color { " << endl;
	fout<<"\t\t\t color [" <<endl;
/*	map<int, int> mymap;
    vector <Point> colors;
//	vector <Point>::iterator it;
	for(int i = 0; i < geom->m_NumTriVerts; i++)
	{
		Point col1(geom->m_TriVertColorsTransparent[4*i+0], geom->m_TriVertColorsTransparent[4*i + 1], geom->m_TriVertColorsTransparent[4*i + 2]);
		if(find(colors.begin(), colors.end(), col1) == colors.end())
			colors.push_back(col1);
		mymap[i] = colors.size()-1;
    }
	for(int i = 0; i<colors.size(); i++)
	fout << "\t\t\t" <<  colors[i] << endl;
	fout<<"\t\t\t ]" << endl;
	fout<<"\t\t }" << endl;
	fout<<"\t\t colorIndex [ ";
	for(map<int, int>::iterator mit = mymap.begin(); mit!= mymap.end(); ++ mit)
	{	if (mit->first %100 ==0)
		fout<<endl;
	 	fout<<mit->second <<" ";
	}	
	fout<<"\n\t\t ]"<<endl;
*/   //Doesn't work. It seems it doesn't support colorPerVertex TRUE.

/*	for(int i = 0; i < geom->m_NumTriVerts; i++)
		fout<<geom->m_TriVertColors[3*i+0]<<" " << geom->m_TriVertColors[3*i+1]<<" " << geom->m_TriVertColors[3*i+2]<< endl;
	fout<<"\t\t\t ]" << endl;
	fout<<"\t\t }"<<endl;
	fout<<"\t\t coordIndex [" << endl;
	for(int i = 0; i < geom->m_NumTris; i++)
		fout<<geom->m_Tris[3*i+0] <<" "<<geom->m_Tris[3*i+1] <<" "<<geom->m_Tris[3*i+2] <<" -1" <<endl;
	fout<<"\t\t ]"<<endl;
	fout<<"\t }" << endl;
	fout<<"}" << endl;
	fout.close(); */

}




// Write out the boundary between inside and outside tetrehedra.
void write_iobdy(const Triangulation& triang, const char* file_prefix)
{
	char filename[100];
	strcat(strcpy(filename, file_prefix), ".io");
	ofstream fout;
	fout.open(filename);
	if (! fout)
	{
		cerr << "Can not open " << filename << " for writing. " << endl;
		exit(1);
	}
	// Count
	int num_facets = 0;
	for (FFI fit = triang.finite_facets_begin();
			fit != triang.finite_facets_end(); ++fit)
	{
		Cell_handle ch = (*fit).first;
		int id = (*fit).second;
		if (ch->outside != ch->neighbor(id)->outside)
		{
			num_facets ++;
		}
	}
	// The header of the output file
	fout <<"OFF";
	fout <<"  " << triang.number_of_vertices(); //The number of points
	fout <<" " << num_facets; //The number of facets
	fout <<" 0" << endl;
	// Write the vertices.
	for (FVI vit = triang.finite_vertices_begin();
			vit != triang.finite_vertices_end(); ++vit)
	{
		fout << vit->point() << endl;
	}
	for (FFI fit = triang.finite_facets_begin();
			fit != triang.finite_facets_end(); ++fit)
	{
		Cell_handle ch = (*fit).first;
		int id = (*fit).second;
		if (ch->outside == ch->neighbor(id)->outside)
		{
			continue;
		}
		fout << " 3\t";
		for (int i = 1; i <= 3; i++)
		{
			fout << " " << ch->vertex((id+i)%4)->id;
		}
		fout << "\t " <<  "1 1 1 1 \n"; ;
	}
	fout.close();
}


// write_axis: writes the medial axis.
void Skel::write_axis(const Triangulation& triang, const int& biggest_medax_comp_id, const char* file_prefix)
{
	char filename[100];
	strcat(strcpy(filename, file_prefix), ".ax");
	ofstream fout;
	fout.open(filename);
	if (! fout)
	{
		cerr << "Can not open " << filename << " for writing. " << endl;
		exit(1);
	}
	char biggest_comp_filename[100];
	strcat(strcpy(biggest_comp_filename, file_prefix), ".00.ax");
	ofstream fout_biggest;
	fout_biggest.open(biggest_comp_filename);
	if (! fout_biggest)
	{
		cerr << "Can not open " << biggest_comp_filename << " for writing. " << endl;
		exit(1);
	}
	fout << "{LIST" << endl;
	fout_biggest << "{LIST" << endl;
	for (FEI eit = triang.finite_edges_begin();
			eit != triang.finite_edges_end(); eit ++)
	{
		Cell_handle c = (*eit).first;
		int uid = (*eit).second, vid = (*eit).third;
		if (is_inf_VF(triang, c, uid, vid))
		{
			continue;    // inf VF
		}
		if (! is_inside_VF(triang, (*eit)))
		{
			continue;    // non-inside VF
		}
		if (is_VF_outside_bounding_box(triang, (*eit), bounding_box))
		{
			continue;    // outside BBOX.
		}
		if (! c->VF_on_medax(uid, vid))
		{
			continue;    // non-medax VF
		}
		CGAL_assertion(c->medax_comp_id[uid][vid] != -1);   // should be in some component.
		// choice of color.
		//srand48(c->medax_comp_id[uid][vid]);
		//double r = drand48(), g = drand48(), b = drand48(), a = 0;
		srand(c->medax_comp_id[uid][vid]);
		double 
		  r = double(rand())/double(RAND_MAX), 
		  g = double(rand())/double(RAND_MAX), 
		  b = double(rand())/double(RAND_MAX),
		  a=0;
		if (c->medax_comp_id[uid][vid] == biggest_medax_comp_id)
		{
			a = 1;
		}
		if (! c->e_tag[uid][vid])
		{
			r = 1;
			g = 0;
			b = 0;
			a = 0.3;
		}
		fout << "# " << c->medax_comp_id[uid][vid] << endl;
		draw_VF(triang, (*eit), r, g, b, a, fout);
		if (c->medax_comp_id[uid][vid] == biggest_medax_comp_id)
		{
			draw_VF(triang, (*eit), r, g, b, a, fout_biggest);
		}
	}
	fout << "}" << endl;
	fout_biggest << "}" << endl;
}

void Skel::write_L_skel(const char* file_prefix) const
{
	char op_filename[100];
	strcat(strcpy(op_filename, file_prefix), ".L");
	ofstream fout;
	fout.open(op_filename);
	fout << "OFF" << endl;
	fout << L.nv << " " << L.ne << " 0" << endl;
	for (int i = 0; i < L.nv; i ++)
	{
		fout << L.vlist[i] << endl;
	}
	for (int i = 0; i < L.ne; i ++)
		fout << "2\t" << L.elist[i].first << " "
			 << L.elist[i].second << " 1 0 0 1" << endl;
}

void Skel::write_u1_skel(const char* file_prefix) const
{
	char op_filename[100];
	strcat(strcpy(op_filename, file_prefix), ".U1");
	ofstream fout;
	fout.open(op_filename);
	fout << "{LIST" << endl;
	for (int i = 0; i < (int)get_nf(); i ++)
	{
		fout << "# " << face_list[i].comp_id << endl;
		//srand48(face_list[i].comp_id);
		srand(face_list[i].comp_id);
		vector<Point> plist;
		for (int j = 0; j < face_list[i].v_cnt; j ++)
		{
			plist.push_back(vert_list[face_list[i].get_vertex(j)].point());
		}
		//draw_poly(plist, drand48(), drand48(), drand48(), 1, fout);
		draw_poly(plist, 
			  double(rand())/double(RAND_MAX), 
			  double(rand())/double(RAND_MAX), 
			  double(rand())/double(RAND_MAX), 
			  1, fout);
	}
	fout << "}" << endl;
	fout.close();
}

void Skel::write_beta(const char* file_prefix) const
{
	char op_filename[100];
	strcat(strcpy(op_filename, file_prefix), ".BETA");
	ofstream fout;
	fout.open(op_filename);
	fout << "{LIST" << endl;
	for (int i = 0; i < (int)get_nf(); i ++)
	{
	  //srand48(face_list[i].comp_id);
		srand(face_list[i].comp_id);
		vector<Point> plist;
		for (int j = 0; j < face_list[i].v_cnt; j ++)
		{
			plist.push_back(vert_list[face_list[i].get_vertex(j)].point());
		}
		if (! comp_pl[i])
		{
			draw_poly(plist, 
				  double(rand())/double(RAND_MAX), 
				  double(rand())/double(RAND_MAX), 
				  double(rand())/double(RAND_MAX), 
				  //drand48(), drand48(), drand48(), 
				  0.1, fout);
		}
		else
		{
			if (face_list[i].beta)
			{
				draw_poly(plist, 
					  double(rand())/double(RAND_MAX), 
					  double(rand())/double(RAND_MAX), 
					  double(rand())/double(RAND_MAX), 
					  //drand48(), drand48(), drand48(), 
					  1, fout);
			}
			else
			{
				draw_poly(plist, 
					  double(rand())/double(RAND_MAX), 
					  double(rand())/double(RAND_MAX), 
					  double(rand())/double(RAND_MAX), 
					  //drand48(), drand48(), drand48(), 
					  0.1, fout);
			}
		}
	}
	fout << "}" << endl;
	fout.close();
}

void Skel::write_u2_skel(const char* file_prefix) const
{
	char op_filename[100];
	strcat(strcpy(op_filename, file_prefix), ".U2");
	ofstream fout;
	fout.open(op_filename);
	fout << "{LIST" << endl;
	for (int i = 0; i < get_ne(); i ++)
	{
		if (! edge_list[i].on_u2)
		{
			continue;
		}
		double r = 1, g = 1, b = 1, a = 1;
		if (edge_list[i].num_inc_face > 1)
		{
			r = 1;
			g = 1;
			b = 0;
			a = 0.3;
		}
		else if (edge_list[i].num_inc_face == 1)
		{
			r = 1;
			g = 0;
			b = 1;
			a = 0.3;
		}
		else
		{
			// if both endpoints are from the same component, color them blue.
			if (in_same_comp(edge_list[i].get_endpoint(0),
							 edge_list[i].get_endpoint(1)))
			{
				r = 0;
				g = 0;
				b = 1;
				a = 0.75;
			}
			else
			{
				r = 1;
				g = 0;
				b = 0;
				a = 1;
			}
		}
		int vid[2] = { edge_list[i].get_endpoint(0),
					   edge_list[i].get_endpoint(1)
					 };
		Point p[2] = { vert_list[vid[0]].point(),
					   vert_list[vid[1]].point()
					 };
		draw_segment(Segment(p[0],p[1]), r,g,b,a, fout);
	}
	fout << "}" << endl;
}

void Skel::write_skel(const char* file_prefix) const
{
	char op_filename[100];
	strcat(strcpy(op_filename, file_prefix), ".skel");
	ofstream fout;
	fout.open(op_filename);
	fout << "{LIST" << endl;
	for (int i = 0; i < (int)get_nf(); i ++)
	{
		fout << "# " << face_list[i].comp_id << endl;
		//srand48(face_list[i].comp_id);
		srand(face_list[i].comp_id);
		vector<Point> plist;
		for (int j = 0; j < face_list[i].v_cnt; j ++)
		{
			plist.push_back(vert_list[face_list[i].get_vertex(j)].point());
		}
		draw_poly(plist, 
			  double(rand())/double(RAND_MAX), 
			  double(rand())/double(RAND_MAX), 
			  double(rand())/double(RAND_MAX), 
			  //drand48(), drand48(), drand48(), 
			  1, fout);
	}
	for (int i = 0; i < get_ne(); i ++)
	{
		double r = 1, g = 1, b = 1, a = 1;
		if (edge_list[i].num_inc_face > 1)
		{
			r = 1;
			g = 1;
			b = 0;
			a = 0.3;
		}
		else if (edge_list[i].num_inc_face == 1)
		{
			r = 1;
			g = 0;
			b = 1;
			a = 0.3;
		}
		else
		{
			CGAL_assertion(edge_list[i].num_inc_face == 0);
			// if both endpoints are from the same component, color them blue.
			if (in_same_comp(edge_list[i].get_endpoint(0),
							 edge_list[i].get_endpoint(1)))
			{
				r = 0;
				g = 0;
				b = 1;
				a = 0.75;
			}
			else
			{
				r = 1;
				g = 0;
				b = 0;
				a = 1;
			}
		}
		int vid[2] = { edge_list[i].get_endpoint(0),
					   edge_list[i].get_endpoint(1)
					 };
		Point p[2] = { vert_list[vid[0]].point(),
					   vert_list[vid[1]].point()
					 };
		fout << "# " << vid[0] << " -> " << vert_list[vid[0]].num_inc_face << endl;
		fout << "# " << vid[1] << " -> " << vert_list[vid[1]].num_inc_face << endl;
		draw_segment(Segment(p[0],p[1]), r,g,b,a, fout);
	}
	for (int i = 0; i < comp_cnt; i ++)
	{
		if (comp_pl[i])
		{
			continue;
		}
		fout << "{appearance {linewidth 4}" << endl;
		fout << "OFF" << endl;
		fout << (int)star[i].size()+1 << " "
			 << (int)star[i].size() << " 0" << endl;
		fout << C[i] << endl;
		for (int j = 0; j < (int)star[i].size(); j ++)
		{
			fout << vert_list[star[i][j]].point() << endl;
		}
		for (int j = 0; j < (int)star[i].size(); j ++)
		{
			fout << "2\t0 " << j+1 << " 0 1 0 1" << endl;
		}
		fout << "}" << endl;
	}
	fout << "}" << endl;
}

void vectors_to_tri_geometry(const std::vector<float>& vertices, const std::vector<unsigned int>& indices, const std::vector<float>& colors, cvcraw_geometry::cvcgeom_t* geom)
{
	assert((vertices.size() % 3) == 0);
	assert((indices.size() % 3) == 0);
	const int finaltris = indices.size()/3;
	const int finalverts = vertices.size()/3;
//	geom->AllocateTris(finalverts, finaltris);
//	geom->AllocateTriVertColors();
    cvcraw_geometry::cvcgeom_t::point_t newVertex;
	cvcraw_geometry::cvcgeom_t::triangle_t newTri;
	cvcraw_geometry::cvcgeom_t::color_t newColor;

	// copy tri verts
	for (int i=0; i<vertices.size()/3; i++)
	{
		newVertex[0] = vertices[3*i+0];
		newVertex[1] = vertices[3*i+1];
		newVertex[2] = vertices[3*i+2];
        geom->points().push_back(newVertex);
	}
	// copy tri indices
	for (int i=0; i<indices.size()/3; i++)
	{
	    newTri[0] = indices[3*i+0];
	    newTri[1] = indices[3*i+1];
	    newTri[2] = indices[3*i+2];

		geom->triangles().push_back(newTri);
	}
	// color
	assert(finalverts == (colors.size()/3));
	for (int i=0; i<colors.size()/3; i++)
	{
		newColor[0] = colors[3*i+0];
		newColor[1] = colors[3*i+1];
		newColor[2] = colors[3*i+2];

		geom->colors().push_back(newColor);
	}
}

void vectors_to_line_geometry(const std::vector<float>& vertices, const std::vector<unsigned int>& indices, float r, float g, float b, float a,  cvcraw_geometry::cvcgeom_t* geom)
{
	assert(vertices.size()%3 == 0);
	assert(indices.size()%2 == 0);
	const int nLineVerts = vertices.size()/3;
	const int nLines = indices.size()/2;
	const int nIndices = indices.size();
    cvcraw_geometry::cvcgeom_t::point_t newVertex;
	cvcraw_geometry::cvcgeom_t::line_t newLine;
	cvcraw_geometry::cvcgeom_t::color_t newColor;

//	geom->AllocateLines(nLineVerts, nLines);
//	geom->AllocateLineColors();
//	assert(geom->points().size() == nLineVerts);
//	assert(geom->lines().size() == nLines);
	// copy into geom
	for (int i=0; i<vertices.size()/3; i++)
	{
		newVertex[0] = vertices[3*i+0];
		newVertex[1] = vertices[3*i+1];
		newVertex[2] = vertices[3*i+2];
        geom->points().push_back(newVertex);
	}
//	geom->m_UniqueLineColors[0] = 1;
//	geom->m_UniqueLineColors[1] = 0;
//	geom->m_UniqueLineColors[2] = 0;
	for (int i=0; i<nLineVerts/3; i++)
	{
	   	newColor[0] = r;
		newColor[1] = g;
		newColor[2] = b;

		geom->colors().push_back(newColor);
	}
	for (int i=0; i<indices.size()/2; i++)
	{   
		newLine[0] = indices[2*i+0];
	    newLine[1] = indices[2*i+1];

		geom->lines().push_back(newLine);
	}
}

void vectors_to_point_geometry(const std::vector<float>& vertices, const std::vector<float>& colors, cvcraw_geometry::cvcgeom_t* geom)
{
	assert(vertices.size()%3 == 0);
	assert(vertices.size() == colors.size());
	const int nPoints = vertices.size()/3;
    cvcraw_geometry::cvcgeom_t::point_t newVertex;
	cvcraw_geometry::cvcgeom_t::color_t newColor;
	// copy into geom

	assert(nPoints == (colors.size()/3));
	for (int i=0; i<vertices.size()/3; i++)
	{
		newVertex[0] = vertices[3*i+0];
		newVertex[1] = vertices[3*i+1];
		newVertex[2] = vertices[3*i+2];
        geom->points().push_back(newVertex);
	}
	for (int i=0; i<colors.size()/3; i++)
	{
		newColor[0] = colors[3*i+0];
		newColor[1] = colors[3*i+1];
		newColor[2] = colors[3*i+2];

		geom->colors().push_back(newColor);
	}

}
