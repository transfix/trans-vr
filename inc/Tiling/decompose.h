#ifndef DECOMPOSE_H
#define DECOMPOSE_H

namespace Tiling {

void free_polygons_ary(PolygonStruct **);

void decompose_sub(short *, int);

PolygonStruct **decompose(VertType *, short *, int, double);

}; // namespace Tiling

#endif
