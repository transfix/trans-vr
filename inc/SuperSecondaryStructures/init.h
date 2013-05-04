#ifndef INIT_H
#define INIT_H

#include <SuperSecondaryStructures/datastruct.h>

namespace SuperSecondaryStructures
{

void
initialize(Triangulation &triang);
void
compute_voronoi_vertex_and_cell_radius(Triangulation &triang);
}
#endif // INIT_H

