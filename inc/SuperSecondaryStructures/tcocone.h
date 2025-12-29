#ifndef TCOCONE_H
#define TCOCONE_H

#include <SuperSecondaryStructures/datastruct.h>
#include <SuperSecondaryStructures/robust_cc.h>
#include <SuperSecondaryStructures/util.h>

namespace SuperSecondaryStructures {
void compute_poles(Triangulation &triang);

void mark_flat_vertices(Triangulation &triang, double ratio,
                        double cocone_phi, double flat_phi);

void tcocone(const double DEFAULT_ANGLE, const double DEFAULT_SHARP,
             const double DEFAULT_FLAT, const double DEFAULT_RATIO,
             Triangulation &triang);

} // namespace SuperSecondaryStructures

#endif // TCOCONE_H
