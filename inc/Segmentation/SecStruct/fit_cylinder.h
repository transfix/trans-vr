#ifndef CYLINDER_H
#define CYLINDER_H

#include <vector>
#include <Segmentation/SecStruct/datastruct.h>

std::vector<std::vector<Point> >
fit_cylinder(const std::vector< Cylinder >& cyls);

#endif // CYLINDER_H

