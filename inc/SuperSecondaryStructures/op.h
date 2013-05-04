#ifndef OP_H
#define OP_H

#include <SuperSecondaryStructures/datastruct.h>
#include <SuperSecondaryStructures/hfn_util.h>

namespace SuperSecondaryStructures
{

void
draw_ray(const Ray& ray,
         const double& r, 
         const double& g, 
         const double& b, 
         const double& a, 
         ofstream& fout);

void
draw_segment(const Segment& segment, 
             const double& r, 
             const double& g, 
             const double& b, 
             const double& a,
             ofstream& fout);


void
draw_poly(const vector<Point>& poly,
          const double& r, 
          const double& g, 
          const double& b, 
          const double& a,
          ofstream& fout);


void
draw_VF(const Triangulation& triang,
        const Edge& dual_e, 
        const double& r, 
        const double& g, 
        const double& b, 
        const double& a,
        ofstream& fout);

void
draw_tetra(const Cell_handle& cell,
           const double& r, 
           const double& g, 
           const double& b, 
           const double& a,
           ofstream& fout);


void 
write_wt( const Triangulation &triang,
	  const char* file_prefix);
void
write_smax(const Triangulation &triang, 
	  map<int, cell_cluster> &cluster_set,
	  const vector<int> &sorted_cluster_index_vector,
	  const int output_seg_count,
	  const char* file_prefix);


}
#endif // OP_H

