#ifndef SWEETMESH_TRIANGLE_H
#define SWEETMESH_TRIANGLE_H

#include <algorithm>
#include <math.h>

#include <SweetMesh/vertex.h>

namespace sweetMesh{

class triangle{
public:
  sweetMesh::sweetMeshVertex vertex0, vertex1, vertex2;
  
  triangle(double v0x, double v0y, double v0z, double v1x, double v1y, double v1z, double v2x, double v2y, double v2z){
    vertex0.set(v0x, v0y, v0z);
    vertex1.set(v1x, v1y, v1z);
    vertex2.set(v2x, v2y, v2z);
  }
  triangle(sweetMeshVertex v0, sweetMeshVertex v1, sweetMeshVertex v2){
    vertex0 = v0;
    vertex1 = v1;
    vertex2 = v2;
  }
  ~triangle() {}
  
  double aspectRatio();
};

}

#endif
