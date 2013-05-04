#ifndef MEDAX_H
#define MEDAX_H

#include <SuperSecondaryStructures/datastruct.h>
#include <SuperSecondaryStructures/util.h>

namespace SuperSecondaryStructures
{

void
compute_medial_axis(Triangulation &triang,
		    const double theta, const double ratio);

};
#endif // MEDAX_H
