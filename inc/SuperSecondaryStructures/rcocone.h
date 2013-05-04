#ifndef RCOCONE_H
#define RCOCONE_H

#include <SuperSecondaryStructures/datastruct.h>
#include <SuperSecondaryStructures/util.h>
#include <SuperSecondaryStructures/robust_cc.h>

namespace SuperSecondaryStructures
{

void
robust_cocone(const double bb_ratio,
	      const double theta_ff,
	      const double theta_if,
	      Triangulation &triang,
	      const char *outfile_prefix);
std::vector<Point> robust_cocone(const double bb_ratio,
				  const double theta_ff,
				  const double theta_if,
				  Triangulation &triang);
}

#endif // RCOCONE_H

