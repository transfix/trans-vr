#include <SweetMesh/tetrahedra.h>

double sweetMesh::tetrahedron::aspectRatio(){
  double alpha;
  double inradius;
  double maxEdge;
  
/*
 * 		[ v0.x v0.y v0.z 1 ]
 * alpha = det	[ v1.x v1.y v1.z 1 ]
 * 		[ v2.x v2.y v2.z 1 ]
 * 		[ v3.x v3.y v3.z 1 ]
 * I'm implementing this determinant using Laplace's formula
 */
  alpha = -1*det(v1, v2, v3) + det(v0, v2, v3) - det(v0, v1, v3) + det(v0, v1, v2);

  //The inradius is abs(alpha) / (sum of the lengths of the (non-unit) normal vectors of each triangle)
  inradius = std::abs(alpha) / ( (v0-v1).crossProduct(v2-v0).euclidianNorm() + (v1-v0).crossProduct(v3-v1).euclidianNorm() + (v2-v0).crossProduct(v3-v0).euclidianNorm() + (v2-v1).crossProduct(v3-v2).euclidianNorm() );
  
  maxEdge = std::max( (v0-v1).euclidianNorm(), std::max( (v1-v2).euclidianNorm(), std::max( (v2-v1).euclidianNorm(), std::max( (v3-v0).euclidianNorm(), std::max( (v3-v1).euclidianNorm(), (v3-v2).euclidianNorm() ) ) ) ) );
  
  return (maxEdge / (2*sqrt(6)*inradius));
}

//   std::cout << "v0:\t" << v0.X() << "\t" << v0.Y() << "\t" << v0.Z() << "\n";
//   std::cout << "v1:\t" << v1.X() << "\t" << v1.Y() << "\t" << v1.Z() << "\n";
//   std::cout << "v2:\t" << v2.X() << "\t" << v2.Y() << "\t" << v2.Z() << "\n";
//   std::cout << "v3:\t" << v3.X() << "\t" << v3.Y() << "\t" << v3.Z() << "\n";
//   std::cout << "det(1,2,3): " << det(v1, v2, v3) << "\tdet(0,2,3): " << det(v0, v2, v3) << "\tdet(0,1,3): " << det(v0, v1, v3) << "\tdet(0,1,2): " << det(v0, v1, v2) << "\nAlpha: " << alpha << "\n";