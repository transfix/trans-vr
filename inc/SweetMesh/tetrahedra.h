#ifndef SWEETMESH_TETRAHEDRA_H
#define SWEETMESH_TETRAHEDRA_H

#include <SweetMesh/vertex.h>
#include <cmath>

namespace sweetMesh{

class tetrahedron{
public:
  sweetMesh::sweetMeshVertex v0, v1, v2, v3;
  
  tetrahedron(sweetMeshVertex V0, sweetMeshVertex V1, sweetMeshVertex V2, sweetMeshVertex V3){
    v0 = V0;
    v1 = V1;
    v2 = V2;
    v3 = V3;
  }
  ~tetrahedron() {}
  
  double aspectRatio();
};

inline double det(double a, double b, double c, double d, double e, double f, double g, double h, double i){
  return a*e*i + b*f*g + c*d*h - a*f*h - b*d*i - c*e*g;
}

}

#endif
