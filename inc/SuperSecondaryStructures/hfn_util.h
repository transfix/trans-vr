#ifndef HFN_UTIL_H
#define HFN_UTIL_H

#include <SuperSecondaryStructures/datastruct.h>
#include <SuperSecondaryStructures/intersect.h>
#include <SuperSecondaryStructures/util.h>

namespace SuperSecondaryStructures {

bool is_maxima(const Cell_handle &c);

bool is_outflow(const Facet &f);

bool is_transversal_flow(const Facet &f);

bool find_acceptor(const Cell_handle &c, const int &id, int &uid, int &vid,
                   int &wid);

bool is_acceptor_for_any_VE(const Triangulation &triang, const Edge &e);

bool is_i2_saddle(const Facet &f);

bool is_i1_saddle(const Edge &e, const Triangulation &triang);

void grow_maxima(Triangulation &triang, Cell_handle c_max);

void find_flow_direction(Triangulation &triang);
} // namespace SuperSecondaryStructures

#endif // HFN_UTIL_H
