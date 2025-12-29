#include "PartitionSurfaceExtractor.h"

#include "Mesh.h"
#include "cvcraw_geometry.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

typedef cvcraw_geometry::geometry_t::point_t point_t;
typedef cvcraw_geometry::geometry_t::triangle_t triangle_t;

PartitionSurfaceExtractor::PartitionSurfaceExtractor(VolMagick::Volume &vol) {
  v = vol;
  xDim = v.XDim();
  yDim = v.YDim();
  zDim = v.ZDim();
  xSpan = (v.XMax() - v.XMin()) / xDim;
  ySpan = (v.YMax() - v.YMin()) / yDim;
  zSpan = (v.ZMax() - v.ZMin()) / zDim;

  // This needs to go into the constructor
  faceIdx[0][0] = 0;
  faceIdx[0][1] = 1;
  faceIdx[0][2] = 2;
  faceIdx[0][3] = 3;
  faceIdx[1][0] = 1;
  faceIdx[1][1] = 2;
  faceIdx[1][2] = 6;
  faceIdx[1][3] = 5;
  faceIdx[2][0] = 4;
  faceIdx[2][1] = 5;
  faceIdx[2][2] = 6;
  faceIdx[2][3] = 7;
  faceIdx[3][0] = 3;
  faceIdx[3][1] = 0;
  faceIdx[3][2] = 4;
  faceIdx[3][3] = 7;
  faceIdx[4][0] = 0;
  faceIdx[4][1] = 1;
  faceIdx[4][2] = 5;
  faceIdx[4][3] = 4;
  faceIdx[5][0] = 3;
  faceIdx[5][1] = 2;
  faceIdx[5][2] = 6;
  faceIdx[5][3] = 7;
}

// Used to initialize various indices used in the code for checking/processing
// faces and voxels
void PartitionSurfaceExtractor::initializeIndices(int i, int j, int k,
                                                  double vtx[8][3],
                                                  int vtxIdx[8][3],
                                                  int voxelIdx[6][3]) {
  // Should be put into a seaparate function
  vtx[0][0] = xSpan * i;
  vtx[0][1] = ySpan * j;
  vtx[0][2] = zSpan * k;
  vtx[1][0] = xSpan * (i + 1);
  vtx[1][1] = ySpan * j;
  vtx[1][2] = zSpan * k;
  vtx[2][0] = xSpan * (i + 1);
  vtx[2][1] = ySpan * (j + 1);
  vtx[2][2] = zSpan * k;
  vtx[3][0] = xSpan * i;
  vtx[3][1] = ySpan * (j + 1);
  vtx[3][2] = zSpan * k;
  vtx[4][0] = xSpan * i;
  vtx[4][1] = ySpan * j;
  vtx[4][2] = zSpan * (k + 1);
  vtx[5][0] = xSpan * (i + 1);
  vtx[5][1] = ySpan * j;
  vtx[5][2] = zSpan * (k + 1);
  vtx[6][0] = xSpan * (i + 1);
  vtx[6][1] = ySpan * (j + 1);
  vtx[6][2] = zSpan * (k + 1);
  vtx[7][0] = xSpan * i;
  vtx[7][1] = ySpan * (j + 1);
  vtx[7][2] = zSpan * (k + 1);

  // SHould be put into a separate function
  vtxIdx[0][0] = i;
  vtxIdx[0][1] = j;
  vtxIdx[0][2] = k;
  vtxIdx[1][0] = (i + 1);
  vtxIdx[1][1] = j;
  vtxIdx[1][2] = k;
  vtxIdx[2][0] = (i + 1);
  vtxIdx[2][1] = (j + 1);
  vtxIdx[2][2] = k;
  vtxIdx[3][0] = i;
  vtxIdx[3][1] = (j + 1);
  vtxIdx[3][2] = k;
  vtxIdx[4][0] = i;
  vtxIdx[4][1] = j;
  vtxIdx[4][2] = (k + 1);
  vtxIdx[5][0] = (i + 1);
  vtxIdx[5][1] = j;
  vtxIdx[5][2] = (k + 1);
  vtxIdx[6][0] = (i + 1);
  vtxIdx[6][1] = (j + 1);
  vtxIdx[6][2] = (k + 1);
  vtxIdx[7][0] = i;
  vtxIdx[7][1] = (j + 1);
  vtxIdx[7][2] = (k + 1);

  // Should be put into a separate function
  voxelIdx[0][0] = i;
  voxelIdx[0][1] = j;
  voxelIdx[0][2] = k - 1;
  voxelIdx[1][0] = i + 1;
  voxelIdx[1][1] = j;
  voxelIdx[1][2] = k;
  voxelIdx[2][0] = i;
  voxelIdx[2][1] = j;
  voxelIdx[2][2] = k + 1;
  voxelIdx[3][0] = i - 1;
  voxelIdx[3][1] = j;
  voxelIdx[3][2] = k;
  voxelIdx[4][0] = i;
  voxelIdx[4][1] = j - 1;
  voxelIdx[4][2] = k;
  voxelIdx[5][0] = i;
  voxelIdx[5][1] = j + 1;
  voxelIdx[5][2] = k;
}

void PartitionSurfaceExtractor::computePartitionSurface(void) {
  // Computes partition surface and populates faceList, vertexList and
  // vertexIndexMap

  // Go through each of the voxels
  for (int i = 0; i < xDim - 1; i++)
    for (int j = 0; j < yDim - 1; j++)
      for (int k = 0; k < zDim - 1; k++) {
        //	if(i%64==0&&j%64==0&&k%64==0)
        //		cout<<i<<" "<<j<<" "<<k<<endl;
        // Define the eight vertices.
        /*
         *      7------6
         *     /|     /|
         *    / |    / |
         *   4------5  |
         *   |  3---|--2
         *   | /    | /
         *   0------1
         *
         *   z
         *   yx
         */

        // The actual vertex coordinates
        double vtx[8][3];
        // The vertex indices; these will always be vtx_x/xspan etc.
        int vtI[8][3];
        // The indices into the voxel with which the i,j,kth voxel is to be
        // compared.
        int vxI[6][3];
        // cout<<"Grid point "<<i<<" "<<j<<" "<<k<<endl;
        // Initialize these indices with proper values
        initializeIndices(i, j, k, vtx, vtI, vxI);
        // cout<<"Aft initIndices"<<endl;
        //  Go through the six voxels around the vertex
        for (int p = 0; p < 6; p++) {
          // cout<<"Checking voxel "<<p<<":";
          //  Check corner cases on the zero side
          if (vxI[p][0] > 0 && vxI[p][1] > 0 && vxI[p][2] > 0) {
            // Check if the voxels are different
            if (v(vxI[p][0], vxI[p][1], vxI[p][2]) != v(i, j, k)) {
              // cout<<" Boundary voxel! "<<endl;
              //  The vector to hold the face which is a boundary face. This
              //  will also result in duplicate faces! Needs to be handled.
              vector<unsigned int> face;
              // Loop through the face index of p corresponding to the
              // interface between vxI[p] and i,j,k
              for (int f = 0; f < 4; f++) {
                // Create a string out of the vertex index vtI for the vertex
                // Index stored in faceIdx[p][f] To use as key
                int currVtxIdx = faceIdx[p][f];
                stringstream s;
                string key;
                s << vtI[currVtxIdx][0] << "-" << vtI[currVtxIdx][1] << "-"
                  << vtI[currVtxIdx][2];
                key = s.str();
                // cout<<"Vertex key: "<<key<<": ";
                //  Check if key exists in the map vertexIndexMap
                if (vertexIndexMap.count(key)) {
                  // cout<<"Vertex already exists."<<endl;
                  // It exists; we only need to add the face to the face
                  // vector.
                  face.push_back(vertexIndexMap[key]);
                  // cout<<"Afte wi"<<endl;
                } else {
                  // cout<<"Vertex does not exist. Adding"<<endl;
                  //  Does not exist; push the vertex into the vertexList
                  vector<float> vertex;
                  vertex.push_back(vtx[currVtxIdx][0]);
                  vertex.push_back(vtx[currVtxIdx][1]);
                  vertex.push_back(vtx[currVtxIdx][2]);
                  // Since we are ging to push the vertex to the end of the
                  // list
                  vertexIndexMap[key] = vertexList.size();
                  // Add the vertex index into the array.
                  face.push_back(vertexList.size());
                  // Add the vertex to the vertexList
                  vertexList.push_back(vertex);
                }
              }
              // cout<<"Size of face: "<<face.size()<<endl;
              // cout<<"Computed face: ";
              // for(int f=0;f<4;f++) cout<<face[f]<<" ";
              // cout<<endl;
              //  We check if the face already exists in the face list. If it
              //  does, we don't add.
              vector<int> fsort;
              for (int f = 0; f < 4; f++)
                fsort.push_back(face[f]);
              sort(fsort.begin(), fsort.end());

              // Create the key for facelist.
              stringstream s;
              for (int f = 0; f < 4; f++)
                s << fsort[f] << "-";

              // cout<<"Face key: "<<s.str()<<" ";
              //  If it does not exist in the faceIndexMap
              if (faceIndexMap.count(s.str()) == 0) {
                // cout<<"does not exist. We add it."<<endl;
                faceIndexMap[s.str()] = faceList.size();
                // Add the face to the faceList vector.
                faceList.push_back(face);
              }
              // cout<<"Aft adding face to face list."<<endl;
            }
            // else cout<<"Nothing here."<<endl;
          }
          // else cout<<"Nothing here."<<endl;
        }
      }
}

vector<vector<unsigned int>> PartitionSurfaceExtractor::getFaceList(void) {
  return faceList;
}

vector<vector<float>> PartitionSurfaceExtractor::getVertexList(void) {
  return vertexList;
}

void PartitionSurfaceExtractor::exportMesh(MMHLS::Mesh &m) {
  for (int i = 0; i < vertexList.size(); i++)
    m.vertexList.push_back(vertexList[i]);
  for (int i = 0; i < faceList.size(); i++) {
    vector<unsigned int> f1, f2;
    f1.push_back(faceList[i][0]);
    f1.push_back(faceList[i][1]);
    f1.push_back(faceList[i][2]);
    f2.push_back(faceList[i][0]);
    f2.push_back(faceList[i][2]);
    f2.push_back(faceList[i][3]);
    m.faceList.push_back(f1);
    m.faceList.push_back(f2);
  }
}

void PartitionSurfaceExtractor::computePartitionSurface(int matId) {
  // Computes partition surface and populates faceList, vertexList and
  // vertexIndexMap
  cout << "Material ID: " << matId << endl;
  // Go through each of the voxels
  for (int i = 0; i < xDim - 1; i++)
    for (int j = 0; j < yDim - 1; j++)
      for (int k = 0; k < zDim - 1; k++) {
        // if(i%64==0&&j%64==0&&k%64==0)
        //	cout<<i<<" "<<j<<" "<<k<<endl;
        // Define the eight vertices.
        /*
         *      7------6
         *     /|     /|
         *    / |    / |
         *   4------5  |
         *   |  3---|--2
         *   | /    | /
         *   0------1
         *
         *   z
         *   yx
         */

        // The actual vertex coordinates
        double vtx[8][3];
        // The vertex indices; these will always be vtx_x/xspan etc.
        int vtI[8][3];
        // The indices into the voxel with which the i,j,kth voxel is to be
        // compared.
        int vxI[6][3];
        // cout<<"Grid point "<<i<<" "<<j<<" "<<k<<endl;
        // Initialize these indices with proper values
        initializeIndices(i, j, k, vtx, vtI, vxI);
        // cout<<"Aft initIndices"<<endl;
        //  Go through the six voxels around the vertex
        for (int p = 0; p < 6; p++) {
          // cout<<"Checking voxel "<<p<<":";
          //  Check corner cases on the zero side
          if (vxI[p][0] >= 0 && vxI[p][1] >= 0 && vxI[p][2] >= 0) {
            // Check if the voxels are different
            int a = v(i, j, k);
            if (a == 37) {
              int b = 0;
            }
            if (v(i, j, k) == matId &&
                v(vxI[p][0], vxI[p][1], vxI[p][2]) != v(i, j, k)) {
              // cout<<" Boundary voxel! "<<endl;
              //  The vector to hold the face which is a boundary face. This
              //  will also result in duplicate faces! Needs to be handled.
              vector<unsigned int> face;
              // Loop through the face index of p corresponding to the
              // interface between vxI[p] and i,j,k
              for (int f = 0; f < 4; f++) {
                // Create a string out of the vertex index vtI for the vertex
                // Index stored in faceIdx[p][f] To use as key
                int currVtxIdx = faceIdx[p][f];
                stringstream s;
                string key;
                s << vtI[currVtxIdx][0] << "-" << vtI[currVtxIdx][1] << "-"
                  << vtI[currVtxIdx][2];
                key = s.str();
                // cout<<"Vertex key: "<<key<<": ";
                //  Check if key exists in the map vertexIndexMap
                if (vertexIndexMap.count(key)) {
                  // cout<<"Vertex already exists."<<endl;
                  // It exists; we only need to add the face to the face
                  // vector.
                  face.push_back(vertexIndexMap[key]);
                  // cout<<"Afte wi"<<endl;
                } else {
                  // cout<<"Vertex does not exist. Adding"<<endl;
                  //  Does not exist; push the vertex into the vertexList
                  vector<float> vertex;
                  vertex.push_back(vtx[currVtxIdx][0]);
                  vertex.push_back(vtx[currVtxIdx][1]);
                  vertex.push_back(vtx[currVtxIdx][2]);
                  // Since we are ging to push the vertex to the end of the
                  // list
                  vertexIndexMap[key] = vertexList.size();
                  // Add the vertex index into the array.
                  face.push_back(vertexList.size());
                  // Add the vertex to the vertexList
                  vertexList.push_back(vertex);
                }
              }
              // cout<<"Size of face: "<<face.size()<<endl;
              // cout<<"Computed face: ";
              // for(int f=0;f<4;f++) cout<<face[f]<<" ";
              // cout<<endl;
              //  We check if the face already exists in the face list. If it
              //  does, we don't add.
              vector<int> fsort;
              for (int f = 0; f < 4; f++)
                fsort.push_back(face[f]);
              sort(fsort.begin(), fsort.end());

              // Create the key for facelist.
              stringstream s;
              for (int f = 0; f < 4; f++)
                s << fsort[f] << "-";

              // cout<<"Face key: "<<s.str()<<" ";
              //  If it does not exist in the faceIndexMap
              if (faceIndexMap.count(s.str()) == 0) {
                // cout<<"does not exist. We add it."<<endl;
                faceIndexMap[s.str()] = faceList.size();
                // Add the face to the faceList vector.
                faceList.push_back(face);
              }
              // cout<<"Aft adding face to face list."<<endl;
            }
            // else cout<<"Nothing here."<<endl;
          }
          // else cout<<"Nothing here."<<endl;
        }
      }
}

void PartitionSurfaceExtractor::clearAll(void) {
  vertexList.clear();
  faceList.clear();
  vertexIndexMap.clear();
  faceIndexMap.clear();
}

VolMagick::BoundingBox PartitionSurfaceExtractor::getBoundingBox(void) {
  float min[3], max[3];
  min[0] = vertexList[0][0];
  max[0] = vertexList[0][0];
  min[1] = vertexList[0][1];
  max[1] = vertexList[0][1];
  min[2] = vertexList[0][2];
  max[2] = vertexList[0][2];

  for (int i = 0; i < vertexList.size(); i++) {
    if (min[0] > vertexList[i][0])
      min[0] = vertexList[i][0];
    if (min[1] > vertexList[i][1])
      min[1] = vertexList[i][1];
    if (min[2] > vertexList[i][2])
      min[2] = vertexList[i][2];
    if (max[0] < vertexList[i][0])
      max[0] = vertexList[i][0];
    if (max[1] < vertexList[i][1])
      max[1] = vertexList[i][1];
    if (max[2] < vertexList[i][2])
      max[2] = vertexList[i][2];
  }

  VolMagick::BoundingBox b(min[0], min[1], min[2], max[0], max[1], max[2]);
  return b;
}
cvcraw_geometry::geometry_t
PartitionSurfaceExtractor::getCVCRawGeometry(void) {
  cvcraw_geometry::geometry_t geom;
  for (int i = 0; i < vertexList.size(); i++) {
    point_t p = {{vertexList[i][0], vertexList[i][1], vertexList[i][2]}};
    geom.points.push_back(p);
    geom.boundary.push_back(1);
  }

  for (int i = 0; i < faceList.size(); i++) {
    triangle_t t1 = {{faceList[i][0], faceList[i][1], faceList[i][2]}},
               t2 = {{faceList[i][0], faceList[i][2], faceList[i][3]}};
    geom.tris.push_back(t1);
    geom.tris.push_back(t2);
  }

  return geom;
}
