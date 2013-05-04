#ifndef SWEETLBIE_OCTREE_H
#define SWEETLBIE_OCTREE_H

#include <VolMagick/VolMagick.h>
#include <SweetMesh/hexmesh.h>
#include <algorithm>

namespace sweetLBIE{

void generateOctree(sweetMesh::hexMesh& octreeMesh, double step, VolMagick::Volume& vol, double isoval);
bool testOctreeHex(double x, double y, double z, double step, double isoval, VolMagick::Volume& vol);
bool testOctreeStep(double step, double isoval, VolMagick::Volume& vol);
double computeStepSize(VolMagick::Volume& vol, double isoval);
double getOctree(VolMagick::Volume& vol, sweetMesh::hexMesh& octreeMesh, double isoval);

}
#endif
