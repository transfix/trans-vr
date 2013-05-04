#include <stdlib.h>
#include <iostream>

#include <SuperSecondaryStructures/op.h>
#include <SuperSecondaryStructures/util.h>


namespace SuperSecondaryStructures
{

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



//-----------------
// write_watertight
//-----------------
// Write out the boundary between inside and outside tetrehedra as surface.
//-----------------------------------------------------------------------
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

  // Count
  int num_facets = 0;
  for ( FFI fit = triang.finite_facets_begin();
	fit != triang.finite_facets_end(); ++fit) {
    Cell_handle ch = (*fit).first;
    int id = (*fit).second;
    CGAL_assertion(ch->bdy[id] == ch->neighbor(id)->
		   bdy[ch->neighbor(id)->index(ch)] );
    if(ch->cocone_flag(id) )
	    num_facets ++;
  }
  
  // The header of the output file

  fout <<"OFF" << endl;
  fout << triang.number_of_vertices(); //The number of points
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

    if(! ch->cocone_flag(id) )
	    continue;
    CGAL_assertion(ch->bdy[id] && ch->neighbor(id)->bdy[ch->neighbor(id)->index(ch)] );
    
    fout << " 3\t";
    for (int i = 1; i <= 3; i++)
    	fout << " " << ch->vertex((id+i)%4)->id;
    fout << "\t " <<  "1 1 1 0.3 \n"; ;
  }

  fout.close();
}


void
write_smax(const Triangulation &triang,
 	  map<int, cell_cluster> &cluster_set,
	  const vector<int> &sorted_cluster_index_vector,
	  const int output_seg_count,
	  const char* file_prefix)
{
   ofstream fout_seg, fout_seg1;
   char filename[100];


   // While writing the segments create the color-plate.
   // For each segment there will be one color. 
   vector<float> r_vector;
   vector<float> g_vector;
   vector<float> b_vector;

   // make a color plate
   for(unsigned int i = 0; i < sorted_cluster_index_vector.size(); i ++)
   {
// arand: changing this since drand48 is not a standard function
// and not available on MinGW
//	   srand48(sorted_cluster_index_vector[i]);
//	   float r = drand48(), g = drand48(), b = drand48();

	   srand(sorted_cluster_index_vector[i]);
	   float r = my_drand(), g = my_drand(), b = my_drand();
	   r_vector.push_back(r);
	   g_vector.push_back(g);
	   b_vector.push_back(b);
   }


   char* file_suffix = "_seg.off";
   char* file_suffix1 = "_seg.rawc";
   for(int i = 0; i < output_seg_count; i ++)
   {
      if(i >= (int)sorted_cluster_index_vector.size())
      {
         cerr << endl << "The number of segments are less than " << output_seg_count << endl;
	 break;
      }
      if(i >= 100)
      {
         cerr << "more than 100 segments will not be output." << endl;
	 break;
      }
      int cl_id = sorted_cluster_index_vector[i];
      char op_fname[100];
	  char op_fname1[100];
      char extn[10];
      extn[0] = '_'; extn[1] = '0' + i/10; extn[2] = '0' + i%10; extn[3] = '\0';
      strcpy(op_fname, file_prefix);
      strcat(op_fname, extn);
      strcat(op_fname, file_suffix);
      cerr << "file : " << op_fname << endl;
	  strcpy(op_fname1, file_prefix);
	  strcat(op_fname1, extn);
	  strcat(op_fname1, file_suffix1);

      fout_seg.open(op_fname);
	  fout_seg1.open(op_fname1);
      if(! fout_seg)
      {
         cerr << "Error in opening output file " << op_fname << endl;
		 exit(1);
      }
	  if(! fout_seg1)
      {
         cerr << "Error in opening output file " << op_fname1<< endl;
	 	exit(1);
      }

      // write the ith biggest cluster.

      // do facet count.
      int facet_count = 0;
      for(FFI fit = triang.finite_facets_begin();
         fit != triang.finite_facets_end(); fit ++)
      {
         Cell_handle c[2] = {(*fit).first, (*fit).first->neighbor((*fit).second)};
	 if(cluster_set[c[0]->id].find() == 
	    cluster_set[c[1]->id].find()) continue;
	 if(cluster_set[c[0]->id].find() != cl_id &&
	    cluster_set[c[1]->id].find() != cl_id )
            continue;
	 facet_count ++;
      }
      // write header.
      fout_seg << "# " << cl_id << endl;
      fout_seg << "OFF" << endl;
      fout_seg << triang.number_of_vertices() << " " 
	       << facet_count << " 0" << endl;
	  fout_seg1 <<  triang.number_of_vertices() << " " 
	       << facet_count << endl;
  
      // write the vertices.
	  double r1 = r_vector[i],
             g1 = g_vector[i],
             b1 = b_vector[i];

      for(FVI vit = triang.finite_vertices_begin();
         vit != triang.finite_vertices_end(); vit ++)
         {
		 	fout_seg << vit->point() << endl;
	   		fout_seg1<< vit->point() <<" " << r1 << " " <<g1 <<" "<<b1 <<endl;
		 }
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
		{
		fout_seg << (*fit).first->vertex(((*fit).second + j)%4)->id << " ";
	 	fout_seg1 << (*fit).first->vertex(((*fit).second + j)%4)->id << " "; 
		} 
	 fout_seg << r << " " << g << " " << b << " 1" << endl;
	 fout_seg1 <<endl;


      }
      fout_seg.close();
	  fout_seg1.close();



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
      system(off2wrl_command); */
   }
   cerr << endl;
   fout_seg.close();
}

};
