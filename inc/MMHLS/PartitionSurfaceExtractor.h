#ifndef __PARTITION_SURF_EXTRACT
#define __PARTITION_SURF_EXTRACT

#include <MMHLS/Mesh1.h>
#include <VolMagick/VolMagick.h>
#include <cvcraw_geometry/cvcraw_geometry.h>
#include <map>
#include <vector>

using namespace std;
using namespace MMHLS;

class PartitionSurfaceExtractor {
private:
  vector<vector<float>> vertexList;
  map<string, int> vertexIndexMap;
  vector<vector<unsigned int>> faceList;
  map<string, unsigned int> faceIndexMap;
  VolMagick::Volume v;
  int xDim, yDim, zDim;
  double xSpan, ySpan, zSpan;
  // The indices into vtx and vtxIdx for the six faces.
  int faceIdx[6][4];

  void initializeIndices(int i, int j, int k, double vtx[8][3],
                         int vtxIdx[8][3], int voxelIdx[6][3]);

public:
  PartitionSurfaceExtractor(VolMagick::Volume &vol);
  void computePartitionSurface(void);
  vector<vector<unsigned int>> getFaceList(void);
  vector<vector<float>> getVertexList(void);
  void computePartitionSurface(int matId);
  void exportMesh(MMHLS::Mesh &m);
  void clearAll(void);
  cvcraw_geometry::geometry_t getCVCRawGeometry(void);
  VolMagick::BoundingBox getBoundingBox(void);
};

#endif
