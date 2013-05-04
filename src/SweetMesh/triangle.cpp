#include <SweetMesh/triangle.h>

double sweetMesh::triangle::aspectRatio(){
  double edgeA, edgeB, edgeC;
  double k;
  double inradius;
  
  edgeA = (vertex1 - vertex0).euclidianNorm();
  edgeB = (vertex2 - vertex1).euclidianNorm();
  edgeC = (vertex0 - vertex2).euclidianNorm();
  k = (edgeA + edgeB + edgeC)/2.0;
  inradius = sqrt(k*(k-edgeA)*(k-edgeB)*(k-edgeC)) / k;
  return std::max(std::max(edgeA, edgeB), edgeC) / (2*sqrt(3)*inradius);
}