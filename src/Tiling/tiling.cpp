#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <algorithm>
#include <utility>
#include <boost/scoped_array.hpp>
#include <boost/format.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/mem_fn.hpp>
#include <boost/shared_ptr.hpp>

// using GSL library for c-spline interpolation
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>

#include <Tiling/common.h>
#include <Tiling/parse_config.h>
#include <Tiling/contour.h>
#include <Tiling/myutil.h>
#include <Tiling/contour_read.h>

#include <Tiling/tiling.h>

namespace Tiling
{
  extern int MATH_FUNC,DEBUG,SILENT;
  extern int CONTOURS_ONLY;
  extern int FILTER_TYPE;
  extern int FROM_CONTOURS;
  extern int SECURITY_CHECK;
  extern int SAVE_CONTOURS;
  extern int DO_APPROX_ONLY;
  extern int STORE_1ST_PASS;
  extern int CHANGE_Z;
  extern int SEPERATE_CONVEX_FILE;
  extern int ALL_CORRESPOND;
  extern int CHRISTIANSEN_METHOD;
  extern int COUNT_TRIANGLE_ONLY;
  extern int NUMERICALLY_STABLE;
  extern int NO_OUTPUT_FILE;
  extern int MESH_FILE; /* prepare file for mesh generation */

  extern int SAME_LEVEL; /* if 1: only tile between the same level */
  extern int DIFF_LEVEL_FILES; /* if 1: put diff. level into diff. files */ 
  extern int Current_level;
  extern int CALC_VOLUME_ONLY;
  extern int NON_VALID;
  extern int Beg_slice;

  extern double TURBERANCE;
  extern float Scale_factor;
  extern double Tolerance;
  extern int WriteIndividualFileInPoly;
  extern double XSHIFT;
  extern double Read_file_time;
  extern float FIRST_Z;
  extern Interpo_struct Istru;
  extern Name_struct Nstru;
  extern double MERGE_DISTANCE_SQUARE;
  extern float Scale_z;

  extern int STATIC_CHQ;

  extern void clear_all_statics();
  extern void my_clear_contour();

  //this code is essentially the original main function for the tiling code, but adapted for use within VolumeRover
  boost::shared_ptr<Geometry> surfaceFromContour(const SurfRecon::ContourPtr& contour,
						 const VolMagick::VolumeFileInfo& volinfo,
						 const std::string& tmpdir)
  {
    my_clear_contour();

    //    DEBUG=1;
    std::vector<double> vec_x, vec_y, vec_z, vec_t; //used to build the input arrays for gsl_spline_init()
    double t, interval;
    const gsl_interp_type *interp;//gsl_interp_cspline; /* cubic spline with natural boundary conditions */
    gsl_interp_accel *acc_x, *acc_y, *acc_z;
    gsl_spline *spline_x, *spline_y, *spline_z;
    int first_slice=-1, last_slice=-1;

    boost::shared_ptr<Geometry> result;

#ifdef __WINDOWS__
    std::string tmpdirdelim("\\");
#else
    std::string tmpdirdelim("/");
#endif

    std::string prefix(tmpdir + tmpdirdelim + "contour");
    std::string suffix(".pts");

    //clear_all_statics();

    //first create a set of pts files corresponding to the contours
    {
      std::ofstream outfile;

      for(unsigned int i=0; i<volinfo.ZDim(); i++)
	{
	  bool no_contour;

	  outfile.open(boost::str(boost::format("%1%%2%%3%")
				  % prefix
				  % i
				  % suffix).c_str(),
		       std::ios_base::out | std::ios_base::trunc);

	  no_contour = true;
	  {
	    /* set interpolation type */
	    switch(contour->interpolationType())
	      {
	      case 0: interp = gsl_interp_linear; break;
	      case 1: interp = gsl_interp_polynomial; break;
	      case 2: interp = gsl_interp_cspline; break;
	      case 3: interp = gsl_interp_cspline_periodic; break;
	      case 4: interp = gsl_interp_akima; break;
	      case 5: interp = gsl_interp_akima_periodic; break;
	      default: interp = gsl_interp_cspline; break;
	      }

	    //this is slow but there shouldn't be that many curves anyways...
	    for(std::vector<SurfRecon::CurvePtr>::iterator cur = contour->curves().begin();
		cur != contour->curves().end();
		cur++)
	      {
		//if the curve lies on the current slice, then write it's points to the pts file
		//also, ignore curves not on XY slices for now since the tiling library cannot handle them
		if(boost::get<0>(**cur) == i &&
		   boost::get<1>(**cur) == SurfRecon::XY &&
		   !boost::get<2>(**cur).empty()
		   && boost::get<2>(**cur).size() > 1) //the tiler has a problem with 1 point contours
		  {
		    if(first_slice == -1) first_slice = i;
		    if(first_slice != -1) last_slice = i;

		    no_contour = false; //we have output, so dont write a "no_contour" pts file

		    if(interp == gsl_interp_linear)
		      {
			SurfRecon::PointPtrList tmplist(boost::get<2>(**cur));

			//std::copy(boost::get<2>(**cur).begin(),boost::get<2>(**cur).end(),tmplist.begin());
			//if(tmplist.front() == tmplist.back()) tmplist.pop_back();
			SurfRecon::PointPtr pfront = tmplist.front();
			SurfRecon::PointPtr pback = tmplist.back();

			if((fabs(pfront->x() - pback->x()) <= 0.00001) &&
			   (fabs(pfront->y() - pback->y()) <= 0.00001) &&
			   (fabs(pfront->z() - pback->z()) <= 0.00001))
			  tmplist.pop_back();

			/*
			outfile << boost::get<2>(**cur).size() << std::endl;

			for(SurfRecon::PointPtrList::iterator pcur = boost::get<2>(**cur).begin();
			    pcur != boost::get<2>(**cur).end();
			    pcur++)
			  {
			    outfile << (*pcur)->x() << " " << (*pcur)->y() << " " << (*pcur)->z() << std::endl;
			  }
			*/

			outfile << tmplist.size() << std::endl;
			
			for(SurfRecon::PointPtrList::const_iterator pcur = tmplist.begin();
			    pcur != tmplist.end();
			    pcur++)
			  {
			    outfile << (*pcur)->x() << " " << (*pcur)->y() << " " << (*pcur)->z() << std::endl;
			  }
		      }
		    else
		      {
			t = 0.0;
			vec_x.clear(); vec_y.clear(); vec_z.clear(); vec_t.clear();
			for(SurfRecon::PointPtrList::iterator pcur = boost::get<2>(**cur).begin();
			    pcur != boost::get<2>(**cur).end();
			    pcur++)
			  {
			    vec_x.push_back((*pcur)->x());
			    vec_y.push_back((*pcur)->y());
			    vec_z.push_back((*pcur)->z());
			    vec_t.push_back(t); t+=1.0;
			  }
			t-=1.0;
			  
			gsl_spline *test_spline = gsl_spline_alloc(interp, 1000); //this is hackish but how else can i find the min size?
			if(vec_t.size() >= gsl_spline_min_size(test_spline))
			  {
			    double time_i;
			    SurfRecon::PointPtrList tmplist;
			    //int count;
			      
			    acc_x = gsl_interp_accel_alloc();
			    acc_y = gsl_interp_accel_alloc();
			    acc_z = gsl_interp_accel_alloc();
			    spline_x = gsl_spline_alloc(interp, vec_t.size());
			    spline_y = gsl_spline_alloc(interp, vec_t.size());
			    spline_z = gsl_spline_alloc(interp, vec_t.size());
			    gsl_spline_init(spline_x, &(vec_t[0]), &(vec_x[0]), vec_t.size());
			    gsl_spline_init(spline_y, &(vec_t[0]), &(vec_y[0]), vec_t.size());
			    gsl_spline_init(spline_z, &(vec_t[0]), &(vec_z[0]), vec_t.size());
			    
			    //Just sample some number of points for non-linear interpolation for now.
			    interval = 1.0/(1+contour->numberOfSamples());
			    //In the future, sample according to the spline's curvature between 2
			    // control points...

			    //this bothers me... please just calculate the number of points in the future!
			    //for(time_i = 0.0, count=0; time_i<=t; time_i+=interval, count++);
			    //outfile << count << std::endl;
			    
			    for(time_i = 0.0; time_i<=t; time_i+=interval)
			      {
				/*outfile << gsl_spline_eval(spline_x, time_i, acc_x) << " " 
					<< gsl_spline_eval(spline_y, time_i, acc_y) << " " 
					<< gsl_spline_eval(spline_z, time_i, acc_z) << std::endl;*/
				tmplist.push_back(SurfRecon::PointPtr(new SurfRecon::Point(gsl_spline_eval(spline_x, time_i, acc_x),
											   gsl_spline_eval(spline_y, time_i, acc_y),
											   gsl_spline_eval(spline_z, time_i, acc_z))));
			      }
			      
			    /* clean up our mess */
			    gsl_spline_free(spline_x);
			    gsl_spline_free(spline_y);
			    gsl_spline_free(spline_z);
			    gsl_interp_accel_free(acc_x);
			    gsl_interp_accel_free(acc_y);
			    gsl_interp_accel_free(acc_z);

			    SurfRecon::PointPtr pfront = tmplist.front();
			    SurfRecon::PointPtr pback = tmplist.back();

			    //remove the last point if the same as first
			    if((fabs(pfront->x() - pback->x()) <= 0.00001) &&
			       (fabs(pfront->y() - pback->y()) <= 0.00001) &&
			       (fabs(pfront->z() - pback->z()) <= 0.00001))
			      tmplist.pop_back();

			    //actually output the points
			    outfile << tmplist.size() << std::endl;
			    for(SurfRecon::PointPtrList::const_iterator pcur = tmplist.begin();
				pcur != tmplist.end();
				pcur++)
			      outfile << (*pcur)->x() << " " << (*pcur)->y() << " " << (*pcur)->z() << std::endl;
			  }
			gsl_spline_free(test_spline);
		      }
		  }
	      }
	  }
	  
	  if(no_contour)
	    {
	      //std::cerr << "no contour on slice " << i << std::endl;
	      outfile << "1" << std::endl;
	      outfile << "0.0 0.0 " << i*volinfo.ZSpan()+volinfo.ZMin() << std::endl;
	    }
	  

	  outfile.close();
	}
    }

    //create an inline cfg file for the tiling code
    std::string cfg(boost::str(boost::format(//"FROM_CONTOURS:\n"
					     //"NON_VALID:\n"
					     "PREFIX: %1%\n"
					     "SUFFIX: %2%\n"
					     "OUTPUT_FILE_NAME: %3%\n"
					     "SLICE_RANGE: %4%\n"
					     //"DETZ: %5%\n"
					     "MERGE_DISTANCE_SQUARE: 1E-24\n")
			       % prefix
			       % suffix
			       % (prefix + "tmp")
			       % boost::str(boost::format("%1% %2%")
					    /*% int(0)
					      % (volinfo.ZDim()-1)*/
					    /*% (first_slice!=-1 ? first_slice : 0)
					      % (last_slice!=-1 ? last_slice : 0)*/
					    % (first_slice-1>=0 ? first_slice-1 : 0)
					    % (last_slice+1<volinfo.ZDim() ? last_slice+1 : volinfo.ZDim()-1))
			       /*% volinfo.ZSpan()*/)
		    );

			       
    boost::scoped_array<char> cfg_str(new char[cfg.size()]);
    strcpy(cfg_str.get(),cfg.c_str());

    STATIC_CHQ = -100; //later, set it to -10000;

    SAVE_CONTOURS = 1;

    tcp_read_database(cfg_str.get(),&Istru,&Nstru);

    Istru.thres+=0.001123f; /* randomize it to avoid .... */

    if(Istru.detz>0)	CHANGE_Z=1;

    if(MATH_FUNC>4)		MATH_FUNC=0;
    
    if(MATH_FUNC)		fprintf(stderr,"The input data is generated as :");

    if(MATH_FUNC==1)	fprintf(stderr,"sphere\n");

    else if(MATH_FUNC==2)	fprintf(stderr,"cylinder\n");
    else if(MATH_FUNC==3)	fprintf(stderr,"x^2 + y^2 - z^2\n");
    else if(MATH_FUNC==4)	fprintf(stderr,"special testing functions\n");

    if(Istru.epsilon!=0.0)	Tolerance=Istru.epsilon;
    if(Tolerance>1.0)
      {
	fprintf(stderr,"Warning! change epsilon from %lf to 1.0\n",Tolerance);
	Tolerance=1.0;
      }
    else if(Tolerance<0.0)
      {
	fprintf(stderr,"Warning! change epsilon from %lf to 0.5\n",Tolerance);
	Tolerance=0.5;
      }

    printf("output file name: %s\n",Nstru.output_file_name);
    printf("Prefix: %s\n",Nstru.prefix);
    printf("Suffix: %s\n",Nstru.suffix);

    //do tiling!
    Beg_slice = Istru.beg;
    tile_all_from_contours(Nstru.output_file_name,Istru.beg, 
			   Istru.end - Istru.beg, Istru.detz,Nstru.prefix,Nstru.suffix);

    //convert output poly file to a Geometry object
    {
      int num;
      unsigned int count=0;
      double inx, iny, inz;

      std::ifstream infile;
      std::ofstream outfile;
      typedef std::vector<SurfRecon::Point> VerticesVec;
      typedef std::set<SurfRecon::Point> VerticesSet;
      typedef std::vector<VerticesSet::iterator> Polygon;
      typedef std::vector<Polygon> Polygons;
      typedef std::vector<unsigned int> IndexVec;
      VerticesVec vertices_vec;
      VerticesSet vertices_set;
      Polygon polygon;
      Polygons polygons;
      IndexVec indices_vec;
      std::string poly_result(Nstru.output_file_name);
      std::string raw_result(Nstru.output_file_name);
      poly_result += ".poly"; //output file name doesn't have poly attached
      raw_result += ".raw";
      
      infile.open(poly_result.c_str());
      
      while(!infile.eof())
	{
	  infile >> num;
	  polygon.clear();
	  for(int i=0; i<num; i++)
	    {
	      infile >> inx >> iny >> inz;
	      
	      //With DETZ set in the config file, the z value has been set to the slice number multiplied by span..
	      //so we must offset it by the minimum of the volume bounding box
	      //inz += volinfo.ZMin();

	      std::pair<VerticesSet::iterator,bool> result = 
		vertices_set.insert(SurfRecon::Point(inx,iny,inz));
	      polygon.push_back(result.first);
	    }
	  
	  polygons.push_back(polygon);
	}

      infile.close();

      vertices_vec.resize(vertices_set.size());
      std::copy(vertices_set.begin(), vertices_set.end(), vertices_vec.begin());      

      for(Polygons::iterator i = polygons.begin();
	  i != polygons.end();
	  i++)
	if((*i).size() == 3) //we only want triangles... the poly file should have only triangles anyways
	  for(Polygon::iterator j = (*i).begin();
	      j != (*i).end();
	      j++)
	    indices_vec.push_back(std::lower_bound(vertices_vec.begin(),vertices_vec.end(),**j) - vertices_vec.begin());

      // Now we have the vertices and indices
      // Load these in Geometry.
      result.reset(new Geometry());
      //cerr << "Starting to create RawC" << endl;
      result->AllocateTris(vertices_vec.size(), indices_vec.size()/3);
      result->AllocateTriVertColors();
      //cerr << "Allocated tris and triverts" << endl;
      outfile.open(raw_result.c_str(), std::ios_base::out | std::ios_base::trunc);
      outfile << vertices_vec.size() << " " << indices_vec.size()/3 << std::endl;

      for(std::vector<SurfRecon::Point>::iterator i = vertices_vec.begin();
	  i != vertices_vec.end();
	  i++)
	{
	  result->m_TriVerts[3*(i-vertices_vec.begin())+0] = i->x();//CGAL::to_double(vit->point().x());
	  result->m_TriVerts[3*(i-vertices_vec.begin())+1] = i->y();//CGAL::to_double(vit->point().y());
	  result->m_TriVerts[3*(i-vertices_vec.begin())+2] = i->z();//CGAL::to_double(vit->point().z());
	  
	  result->m_TriVertColors[3*(i-vertices_vec.begin())+0] = boost::get<0>(contour->color());
	  result->m_TriVertColors[3*(i-vertices_vec.begin())+1] = boost::get<1>(contour->color());
	  result->m_TriVertColors[3*(i-vertices_vec.begin())+2] = boost::get<2>(contour->color());

	  outfile << i->x() << " " << i->y() << " " << i->z() << std::endl;
	}
      
      for(std::vector<unsigned int>::iterator i = indices_vec.begin();
	  i != indices_vec.end();
	  i++)
	{
	  result->m_Tris[i-indices_vec.begin()] = *i;

	  outfile << *i << " ";

	  if((int(i-indices_vec.begin())+1) % 3 == 0) outfile << std::endl;
	}

      outfile.close();
    }

    return result;
  }

  std::vector<boost::shared_ptr<Geometry> > surfacesFromContours(const SurfRecon::ContourPtrArray& contours, 
								 const VolMagick::VolumeFileInfo& volinfo,
								 unsigned int var, unsigned int time,
								 const std::string& tmpdir)
  {
    std::vector<boost::shared_ptr<Geometry> > geometries;

    for(SurfRecon::ContourPtrMap::const_iterator i = contours[var][time].begin();
	i != contours[var][time].end();
	i++)
      geometries.push_back(surfaceFromContour((*i).second,volinfo,tmpdir));

    return geometries;
  }
};
