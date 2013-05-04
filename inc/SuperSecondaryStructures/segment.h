#ifndef SEGMENT_H
#define SEGMENT_H

#include <SuperSecondaryStructures/datastruct.h>
#include <SuperSecondaryStructures/util.h>
#include <SuperSecondaryStructures/robust_cc.h>

namespace SuperSecondaryStructures
{

vector<int>
segment_shape(Triangulation &triang, const double merge_ratio, 
	      map<int, cell_cluster> &cluster_set);

void
refine_segmentation(const Triangulation &triang,
		    map<int, cell_cluster> &cluster_set,
		    const vector<int> &sorted_cluster_index_vector,
		    const float &refine_threshold,
		    vector<bool> &cluster_reject_vector);
}

#endif // SEGMENT_H

