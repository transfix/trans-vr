#ifndef __MINKOWSKI_H__
#define __MINKOWSKI_H__

CONTOURTILER_BEGIN_NAMESPACE

Polygon_2 offset_minkowski(const Polygon_2 &p, const Number_type radius);

Polygon_2 close_minkowski(const Polygon_2 &p, const Number_type radius);

Polygon_2 open_minkowski(const Polygon_2 &p, const Number_type radius);

CONTOURTILER_END_NAMESPACE

#endif
