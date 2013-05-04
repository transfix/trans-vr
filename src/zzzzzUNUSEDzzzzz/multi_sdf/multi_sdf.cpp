/* $Id: multi_sdf.cpp 4196 2011-06-06 19:01:48Z jsweet $ */

// Samrat
// Started Dec 4th, 2007.

#include <cstdio>
#include <multi_sdf/mesh_io.h>
#include <multi_sdf/sdf.h>
#include <multi_sdf/kdtree.h>
#include <multi_sdf/matrix.h>
#include <multi_sdf/dt.h>
#include <multi_sdf/multi_sdf.h>

using namespace std;

namespace multi_sdf
{

const int DIMX = 100;
const int DIMY = 100;
const int DIMZ = 100;
const double BB_SCALE = 0.2;

void
construct_bbox(const Mesh& mesh, vector<double>& bbox)
{
   double x_min = HUGE, x_max = -HUGE,
          y_min = HUGE, y_max = -HUGE,
          z_min = HUGE, z_max = -HUGE;
   for(int i = 0; i < mesh.get_nv(); i ++)
   {
      if( mesh.vert_list[i].iso() ) continue;
      Point p = mesh.vert_list[i].point();
      
      // check x-span
      if(CGAL::to_double(p.x()) < x_min) 
         x_min = CGAL::to_double(p.x());
      if(CGAL::to_double(p.x()) > x_max) 
         x_max = CGAL::to_double(p.x());
      // check y-span
      if(CGAL::to_double(p.y()) < y_min) 
         y_min = CGAL::to_double(p.y());
      if(CGAL::to_double(p.y()) > y_max) 
         y_max = CGAL::to_double(p.y());
      // check z-span
      if(CGAL::to_double(p.z()) < z_min) 
         z_min = CGAL::to_double(p.z());
      if(CGAL::to_double(p.z()) > z_max) 
         z_max = CGAL::to_double(p.z());
   }
   bbox.push_back(x_min - BB_SCALE*(x_max-x_min));
   bbox.push_back(y_min - BB_SCALE*(y_max-y_min));
   bbox.push_back(z_min - BB_SCALE*(z_max-z_min));

   bbox.push_back(x_max + BB_SCALE*(x_max-x_min));
   bbox.push_back(y_max + BB_SCALE*(y_max-y_min));
   bbox.push_back(z_max + BB_SCALE*(z_max-z_min));
}

void
assign_sdf_weight(Mesh& mesh, vector<double>& weights)
{
   // map the color of each facet to a weight (scalar).
   // for the time being the information is in file called "weights".
   ifstream fin;
   fin.open("weights");
   istream_iterator<double> input(fin);
   istream_iterator<double> beyond;
   double tw = 0;
   while(input != beyond) { tw += *input; weights.push_back(*input); input++; }  
   for(int i = 0; i < (int)weights.size(); i ++) weights[i] /= tw;
}


VolMagick::Volume signedDistanceFunction(const boost::shared_ptr<Geometry>& geom,
					 const VolMagick::Dimension& dim,
					 const VolMagick::BoundingBox& bbox)
{
  int dimx = dim[0], dimy = dim[1], dimz = dim[2];

  // read the annotated input file and
  Mesh mesh;
  cerr << "Reading input mesh ";
  read_labeled_mesh(mesh, geom);
  cerr << "done." << endl;

  // build a bounding box around the input and store the
  // origin, span etc.
  //  vector<double> bbox;
  //  construct_bbox(mesh, bbox);
  VolMagick::BoundingBox box(bbox);
  if(box.isNull())
    {
      geom->GetReadyToDrawWire(); //make sure we have calculated extents for this geometry already
      box[0] = geom->m_Min[0];
      box[1] = geom->m_Min[1];
      box[2] = geom->m_Min[2];
      box[3] = geom->m_Max[0];
      box[4] = geom->m_Max[1];
      box[5] = geom->m_Max[2];
    }

  // construct a kd-tree of all the non-isolated mesh_vertices.
  vector<VECTOR3> points;
  vector<Point> pts;
  for(int i = 0; i < mesh.get_nv(); i ++)
  {
     if( mesh.vert_list[i].iso() ) continue;
     Point p = mesh.vert_list[i].point();
     pts.push_back(p);
     points.push_back(VECTOR3(CGAL::to_double(p.x()),
                              CGAL::to_double(p.y()),
                              CGAL::to_double(p.z())));
  }
  KdTree kd_tree(points, 20);
  kd_tree.setNOfNeighbours(1);

  // Now perform a reconstruction to build a tetrahedralized solid
  // with in-out marked.
  Triangulation triang;
  recon(pts, triang);

  // assign weight to each triangle.
  vector<double> weights;
  // assign_sdf_weight(mesh, weights); // comment out for uniform weight.

  VolMagick::Volume vol;

  cerr << "SDF " << endl;
  try
    {
      vol.dimension(VolMagick::Dimension(dimx,dimy,dimz));
      vol.voxelType(VolMagick::Float);
      vol.boundingBox(box);

      for(unsigned int k=0; k<vol.ZDim(); k++)
      {
	for(unsigned int j=0; j<vol.YDim(); j++)
        {
	  for(unsigned int i=0; i<vol.XDim(); i++)
	    {
              double x = vol.XMin() + i*vol.XSpan();
              double y = vol.YMin() + j*vol.YSpan();
              double z = vol.ZMin() + k*vol.ZSpan();
              double fn_val = sdf(Point(x,y,z), mesh, weights, kd_tree, triang);
	      vol(i,j,k, fn_val);
	    }
        }
	fprintf(stderr,
		"%5.2f %%\r",
		(float(k)/float(vol.ZDim()-1))*100.0);
      }

      vol.desc("multi_sdf");
    }
  catch(VolMagick::Exception &e)
    {
      cerr << e.what() << endl;
    }
  catch(std::exception &e)
    {
      cerr << e.what() << endl;
    }

  cerr << endl << "done." << endl;

  return vol;
}

}
