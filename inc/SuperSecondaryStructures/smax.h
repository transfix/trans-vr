
#ifndef SMAX_H
#define SMAX_H

#include <SuperSecondaryStructures/datastruct.h>
#include <SuperSecondaryStructures/hfn_util.h>
#include <SuperSecondaryStructures/op.h>
#include <SuperSecondaryStructures/robust_cc.h>

namespace SuperSecondaryStructures {
vector<int> compute_smax(Triangulation &triang,
                         map<int, cell_cluster> &cluster_set,
                         const double &mr);
}
#endif
