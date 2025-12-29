#ifndef SWEETLBIE_MESHGENERATION_H
#define SWEETLBIE_MESHGENERATION_H

#include <SweetLBIE/octree.h>
#include <SweetMesh/hexmesh.h>
#include <SweetMesh/volRoverDisplay.h>
#include <VolMagick/VolMagick.h>
#include <cvcraw_geometry/cvcraw_geometry.h>

namespace sweetLBIE {

void orientCorner(std::list<sweetMesh::hexahedron>::iterator &hexItr,
                  std::list<sweetMesh::hexVertex>::iterator &v0,
                  std::list<sweetMesh::hexVertex>::iterator &v1,
                  std::list<sweetMesh::hexVertex>::iterator &v2,
                  std::list<sweetMesh::hexVertex>::iterator &v3,
                  std::list<sweetMesh::hexVertex>::iterator &v4,
                  std::list<sweetMesh::hexVertex>::iterator &v5,
                  std::list<sweetMesh::hexVertex>::iterator &v6,
                  std::list<sweetMesh::hexVertex>::iterator &v7);
void orientEdge(std::list<sweetMesh::hexahedron>::iterator &hexItr,
                std::list<sweetMesh::hexVertex>::iterator &v0,
                std::list<sweetMesh::hexVertex>::iterator &v1,
                std::list<sweetMesh::hexVertex>::iterator &v2,
                std::list<sweetMesh::hexVertex>::iterator &v3,
                std::list<sweetMesh::hexVertex>::iterator &v4,
                std::list<sweetMesh::hexVertex>::iterator &v5,
                std::list<sweetMesh::hexVertex>::iterator &v6,
                std::list<sweetMesh::hexVertex>::iterator &v7);
void orientFace(std::list<sweetMesh::hexahedron>::iterator &hexItr,
                std::list<sweetMesh::hexVertex>::iterator &v0,
                std::list<sweetMesh::hexVertex>::iterator &v1,
                std::list<sweetMesh::hexVertex>::iterator &v2,
                std::list<sweetMesh::hexVertex>::iterator &v3,
                std::list<sweetMesh::hexVertex>::iterator &v4,
                std::list<sweetMesh::hexVertex>::iterator &v5,
                std::list<sweetMesh::hexVertex>::iterator &v6,
                std::list<sweetMesh::hexVertex>::iterator &v7);
void dividePattern_corner(std::list<sweetMesh::hexahedron>::iterator hexItr,
                          sweetMesh::hexMesh &mesh, VolMagick::Volume &vol,
                          double isoval, bool meshLessThanIsoval);
void dividePattern_edge(std::list<sweetMesh::hexahedron>::iterator hexItr,
                        VolMagick::Volume &vol, double isoval,
                        bool meshLessThanIsoval);
void dividePattern_face(std::list<sweetMesh::hexahedron>::iterator hexItr,
                        VolMagick::Volume &vol, double isoval,
                        bool meshLessThanIsoval);
void dividePattern_all(std::list<sweetMesh::hexahedron>::iterator hexItr,
                       VolMagick::Volume &vol, double isoval,
                       bool meshLessThanIsoval);
void subdivideHexes(sweetMesh::hexMesh &mesh, VolMagick::Volume &vol,
                    double isoval, bool meshLessThanIsoval);
// Return true if we change the sign of a vertex, return false if nothing is
// changed.
bool setHexSignsTemplate(std::list<sweetMesh::hexahedron>::iterator hexItr);
bool testEpsilon(std::list<sweetMesh::hexVertex>::iterator vertexItr,
                 VolMagick::Volume &vol, double isoval, double epsilon);
// void setHexSignsInitialPass(std::list<sweetMesh::hexahedron>::iterator
// hexItr, VolMagick::Volume& vol, double isoval, bool meshLessThanIsoval,
// double epsilon);
void setVertexSigns(sweetMesh::hexMesh::hexMesh &mesh, VolMagick::Volume &vol,
                    double isoval, bool meshLessThanIsoval, double epsilon);
// void setScaffoldVerticesAroundCenter(sweetMesh::vertex center, double
// octreeStep, sweetMesh::vertex& v0, sweetMesh::vertex& v1,
// sweetMesh::vertex& v2, sweetMesh::vertex& v3, sweetMesh::vertex& v4,
// sweetMesh::vertex& v5, sweetMesh::vertex& v6, sweetMesh::vertex& v7); bool
// testShouldAddHex(hexMesh::vertex center, double octreeStep, double isoval,
// bool meshLessThanIsoval, VolMagick::Volume& vol);
void generateMesh(sweetMesh::hexMesh &mesh, double octreeStep, double isoval,
                  bool meshLessThanIsoval, VolMagick::Volume &vol);
void LBIE_main(VolMagick::Volume &vol, double isoval,
               bool meshLessThanIsoval);
void test_LBIE(std::string &cur);

} // namespace sweetLBIE

#endif