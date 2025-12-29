#include <SweetMesh/hexmeshtest.h>

/*
//outputFaces()=====================
void outputFaces(sweetMesh::hexMesh& mesh, std::list<sweetMesh::rawc>&
outputRawc){ sweetMesh::rawc triangleA, triangleB, triangleC;
        for(std::list<sweetMesh::hexahedron>::iterator
hexItr=mesh.hexahedra.begin(); hexItr!=mesh.hexahedra.end(); hexItr++){
                std::vector<sweetMesh::quadFace> faces = hexItr->getFaces();
                for(unsigned int n=0; n<6; n++){
                        triangleA.vA = *(faces[n].corners[0].myVertexItr);
                        triangleA.vB = *(faces[n].corners[1].myVertexItr);
                        triangleA.vC = *(faces[n].corners[2].myVertexItr);
                        triangleB.vA = *(faces[n].corners[1].myVertexItr);
                        triangleB.vB = *(faces[n].corners[2].myVertexItr);
                        triangleB.vC = *(faces[n].corners[3].myVertexItr);
                        triangleC.vA = *(faces[n].corners[0].myVertexItr);
                        triangleC.vB = *(faces[n].corners[2].myVertexItr);
                        triangleC.vC = *(faces[n].corners[3].myVertexItr);
                        outputRawc.push_back(triangleA);
                        outputRawc.push_back(triangleB);
                        outputRawc.push_back(triangleC);
                }
        }
}

//outputBoundaryHexes()=============
void outputBoundaryHexes(sweetMesh::hexMesh& mesh,
std::list<sweetMesh::volRover_linec>& outputLines){ sweetMesh::volRover_linec
newLinec; bool print; for(std::list<sweetMesh::hexEdge>::iterator edgeItr =
mesh.edges.begin(); edgeItr!=mesh.edges.end(); edgeItr++){ print = false;
                for(std::list<std::list<sweetMesh::hexahedron>::iterator>::iterator
hexItrItr=edgeItr->adjacentHexItrs.begin();
hexItrItr!=edgeItr->adjacentHexItrs.end(); hexItrItr++){
                        if((*hexItrItr)->hasSurfaceVertex){
                                print = true;
                        }
                }
                if(print){
                        newLinec.startVertex = *edgeItr->vA_Itr;
                        newLinec.startColor.set(GREY);
                        newLinec.endVertex = *edgeItr->vB_Itr;
                        newLinec.endColor.set(GREY);
                        outputLines.push_back(newLinec);
                }
        }
}
//outputHexesWithNonPosJac()========
void outputHexesWithNonPosJac(sweetMesh::hexMesh& mesh,
std::list<sweetMesh::volRover_linec>& outputLines){ sweetMesh::volRover_linec
newLinec; bool print; for(std::list<sweetMesh::hexEdge>::iterator edgeItr =
mesh.edges.begin(); edgeItr!=mesh.edges.end(); edgeItr++){ print = false;
                for(std::list<std::list<sweetMesh::hexahedron>::iterator>::iterator
hexItrItr=edgeItr->adjacentHexItrs.begin();
hexItrItr!=edgeItr->adjacentHexItrs.end(); hexItrItr++){
                        if((*hexItrItr)->hasNonPosHexJacobian){
                                print = true;
                        }
                }
                if(print){
                        newLinec.startVertex = *edgeItr->vA_Itr;
                        newLinec.startColor.set(GREY);
                        newLinec.endVertex = *edgeItr->vB_Itr;
                        newLinec.endColor.set(GREY);
                        outputLines.push_back(newLinec);
                }
        }
}
//outputSurfaceEdges()==============
void outputSurfaceEdges(sweetMesh::hexMesh& mesh,
std::list<sweetMesh::volRover_linec>& outputLines){ sweetMesh::volRover_linec
newLinec; for(std::list<sweetMesh::hexEdge>::iterator edgeItr =
mesh.edges.begin(); edgeItr!=mesh.edges.end(); edgeItr++){
                if(edgeItr->liesOnSurface){
                        newLinec.startVertex = *edgeItr->vA_Itr;
                        newLinec.startColor.set(GREY);
                        newLinec.endVertex = *edgeItr->vB_Itr;
                        newLinec.endColor.set(GREY);
                        outputLines.push_back(newLinec);
                }
        }
}
//ouputQuadsWithVertexNormals()========
void outputSurfaceQuadsWithVertexNormals(sweetMesh::hexMesh& mesh,
std::list<sweetMesh::volRover_linec>& outputLines){ sweetMesh::volRover_linec
newLinec; sweetMesh::vertex normal, e1, e2; double length;
        std::vector<sweetMesh::quadFace> quads;
        for(std::list<sweetMesh::hexahedron>::iterator
hexItr=mesh.hexahedra.begin(); hexItr!=mesh.hexahedra.end(); hexItr++){
                if(hexItr->hasSurfaceVertex){
                        quads = hexItr->getFaces();
                        for(unsigned int n=0; n<6; n++){
                                if(quads[n].liesOnBoundary){
                                        newLinec.startVertex =
*(quads[n].corners[0].myVertexItr); newLinec.startColor.set(GREY);
                                        newLinec.endVertex =
*(quads[n].corners[1].myVertexItr); newLinec.endColor.set(GREY);
                                        outputLines.push_back(newLinec);
                                        quads[n].makeEdgeVectors(0, e1, e2);
                                        length = e1.euclidianNorm();
                                        e1 /= e1.euclidianNorm();
                                        e2 /= e2.euclidianNorm();
                                        normal = e1.crossProduct(e2);
                                        normal -=
*quads[n].corners[0].myVertexItr; normal /= normal.euclidianNorm(); normal =
normal * length; normal += *quads[n].corners[0].myVertexItr;
                                        newLinec.startVertex =
*quads[n].corners[0].myVertexItr; newLinec.startColor.set(GREEN);
                                        newLinec.endVertex = normal;
                                        newLinec.endColor.set(GREEN);
                                        outputLines.push_back(newLinec);

                                        newLinec.startVertex =
*(quads[n].corners[1].myVertexItr); newLinec.startColor.set(GREY);
                                        newLinec.endVertex =
*(quads[n].corners[2].myVertexItr); newLinec.endColor.set(GREY);
                                        outputLines.push_back(newLinec);
                                        quads[n].makeEdgeVectors(1, e1, e2);
                                        length = e1.euclidianNorm();
                                        e1 /= e1.euclidianNorm();
                                        e2 /= e2.euclidianNorm();
                                        normal = e1.crossProduct(e2);
                                        normal -=
*quads[n].corners[1].myVertexItr; normal /= normal.euclidianNorm(); normal =
normal * length; normal += *quads[n].corners[1].myVertexItr;
                                        newLinec.startVertex =
*quads[n].corners[1].myVertexItr; newLinec.startColor.set(RED);
                                        newLinec.endVertex = normal;
                                        newLinec.endColor.set(RED);
                                        outputLines.push_back(newLinec);

                                        newLinec.startVertex =
*(quads[n].corners[2].myVertexItr); newLinec.startColor.set(GREY);
                                        newLinec.endVertex =
*(quads[n].corners[3].myVertexItr); newLinec.endColor.set(GREY);
                                        outputLines.push_back(newLinec);
                                        quads[n].makeEdgeVectors(2, e1, e2);
                                        length = e1.euclidianNorm();
                                        e1 /= e1.euclidianNorm();
                                        e2 /= e2.euclidianNorm();
                                        normal = e1.crossProduct(e2);
                                        normal -=
*quads[n].corners[2].myVertexItr; normal /= normal.euclidianNorm(); normal =
normal * length; normal += *quads[n].corners[2].myVertexItr;
                                        newLinec.startVertex =
*quads[n].corners[2].myVertexItr; newLinec.startColor.set(BLUE);
                                        newLinec.endVertex = normal;
                                        newLinec.endColor.set(BLUE);
                                        outputLines.push_back(newLinec);

                                        newLinec.startVertex =
*(quads[n].corners[3].myVertexItr); newLinec.startColor.set(GREY);
                                        newLinec.endVertex =
*(quads[n].corners[0].myVertexItr); newLinec.endColor.set(GREY);
                                        outputLines.push_back(newLinec);
                                        quads[n].makeEdgeVectors(3, e1, e2);
                                        length = e1.euclidianNorm();
                                        e1 /= e1.euclidianNorm();
                                        e2 /= e2.euclidianNorm();
                                        normal = e1.crossProduct(e2);
                                        normal -=
*quads[n].corners[3].myVertexItr; normal /= normal.euclidianNorm(); normal =
normal * length; normal += *quads[n].corners[3].myVertexItr;
                                        newLinec.startVertex =
*quads[n].corners[3].myVertexItr; newLinec.startColor.set(PURPLE);
                                        newLinec.endVertex = normal;
                                        newLinec.endColor.set(PURPLE);
                                        outputLines.push_back(newLinec);

                                }
                        }
                }
        }
}
//ouputNonPosJacIndicators()===========
void ouputNonPosJacIndicators(sweetMesh::hexMesh& mesh,
std::list<sweetMesh::volRover_linec>& outputLines){ sweetMesh::volRover_linec
newLinec;

        std::list<sweetMesh::hexCorner>::iterator cornerItr;
        for(cornerItr=mesh.hexCorners.begin();
cornerItr!=mesh.hexCorners.end(); cornerItr++){ if(cornerItr->jacobian <= 0){
                        newLinec.startVertex = *cornerItr->myVertexItr;
                        newLinec.endVertex =
*cornerItr->myHexItr->cornerItrs[0]->myVertexItr +
*cornerItr->myHexItr->cornerItrs[1]->myVertexItr +
*cornerItr->myHexItr->cornerItrs[2]->myVertexItr +
*cornerItr->myHexItr->cornerItrs[3]->myVertexItr +
*cornerItr->myHexItr->cornerItrs[4]->myVertexItr +
*cornerItr->myHexItr->cornerItrs[5]->myVertexItr +
*cornerItr->myHexItr->cornerItrs[6]->myVertexItr +
*cornerItr->myHexItr->cornerItrs[7]->myVertexItr; newLinec.endVertex /= 8;
                        newLinec.startColor.set(-cornerItr->jacobian/mesh.minHexJacobian,
0, 1+cornerItr->jacobian/mesh.minHexJacobian);
                        newLinec.endColor.set(-cornerItr->jacobian/mesh.minHexJacobian,
0, 1+cornerItr->jacobian/mesh.minHexJacobian);
                        outputLines.push_back(newLinec);
                }
        }
}

*/

void sweetMeshTest() {
  std::cout << "inside SweetMeshTest()\n";
  std::ifstream instream("meshOriginal.rawhs");
  sweetMesh::hexMesh mesh;
  sweetMesh::readRAWHSfile(mesh, instream);
  instream.close();
  std::cout << "Done reading mesh\n";
  mesh.printStatistics();

  sweetMesh::visualMesh visMesh(mesh);
  visMesh.renderAllEdges = true;
  visMesh.renderAllSurfaceQuads = true;
  visMesh.refresh();

  cvcraw_geometry::write(cvcraw_geometry::geometry_t(visMesh),
                         "meshOriginal.raw");
  std::cout << "Done with SweetMeshTest()\n";
}
