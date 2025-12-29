#ifndef CYLINDER_H
#define CYLINDER_H

#include <Segmentation/SecStruct/datastruct.h>
#include <vector>

std::vector<std::vector<Point>>
fit_cylinder(const std::vector<Cylinder> &cyls);

#endif // CYLINDER_H
