/*
The vertices of the hexahedron should be labeled as follows:
   6
 / |\
7  | 5
|\ |/|
| 4| |
| || |
| |2 |
|/| \|
3 |  1
 \| /
  0      */

#include <SweetMesh/hexmesh.h>

/********************************************************************************/
//	hexMesh
/********************************************************************************/

// addHex()==========================
std::list<sweetMesh::hexahedron>::iterator sweetMesh::hexMesh::addHex(
    hexVertex &v0, hexVertex &v1, hexVertex &v2, hexVertex &v3, hexVertex &v4,
    hexVertex &v5, hexVertex &v6, hexVertex &v7, unsigned int orderIndex) {
  unsigned long long newHandleID;
  hexCorner newHexCorner;
  std::list<hexahedron>::iterator thisHexItr;
  if (deletedHandles.empty()) {
    newHandleID = hexahedra.size();
  } else {
    newHandleID = deletedHandles.front();
    deletedHandles.pop();
  }
  hexahedron newHex(newHandleID);
  newHex.setOrderIndex(orderIndex);
  newHex.cornerItrs.resize(8);
  for (unsigned int n = 0; n < 8; n++) {
    newHex.cornerItrs[n] = hexCorners.insert(hexCorners.end(), newHexCorner);
  }
  vertices.sort();
  newHex.cornerItrs[0]->myVertexItr = addVertex(v0);
  newHex.cornerItrs[1]->myVertexItr = addVertex(v1);
  newHex.cornerItrs[2]->myVertexItr = addVertex(v2);
  newHex.cornerItrs[3]->myVertexItr = addVertex(v3);
  newHex.cornerItrs[4]->myVertexItr = addVertex(v4);
  newHex.cornerItrs[5]->myVertexItr = addVertex(v5);
  newHex.cornerItrs[6]->myVertexItr = addVertex(v6);
  newHex.cornerItrs[7]->myVertexItr = addVertex(v7);

  thisHexItr = hexahedra.insert(hexahedra.end(), newHex);

  for (unsigned int n = 0; n < 8; n++) {
    thisHexItr->cornerItrs[n]->myHexItr = thisHexItr;
    thisHexItr->cornerItrs[n]->myCornerPosition = n;
    thisHexItr->cornerItrs[n]->myVertexItr->hexCornerItrs.push_back(
        thisHexItr->cornerItrs[n]);
  }
  thisHexItr->computeJacobians(minHexJacobian, maxHexJacobian,
                               numNonPosHexJacobians);
  thisHexItr->hasSurfaceVertex = false;
  if (v0.liesOnBoundary || v1.liesOnBoundary || v2.liesOnBoundary ||
      v3.liesOnBoundary || v4.liesOnBoundary || v5.liesOnBoundary ||
      v6.liesOnBoundary || v7.liesOnBoundary) {
    thisHexItr->hasSurfaceVertex = true;
    thisHexItr->displayHex = true;
  } else {
    thisHexItr->displayHex = false;
  }
  createQuads(thisHexItr);
  createEdges(thisHexItr);
  return thisHexItr;
}
// removeHex()=======================
std::list<sweetMesh::hexahedron>::iterator
sweetMesh::hexMesh::removeHex(std::list<hexahedron>::iterator &deleteHexItr) {
  deletedHandles.push(deleteHexItr->HandleID());
  detachHex(*deleteHexItr);
  return --hexahedra.erase(deleteHexItr);
}
#include <boost/concept_check.hpp>
// clear()===========================
void sweetMesh::hexMesh::clear() {
  std::list<hexahedron>::iterator hexItr = hexahedra.begin();
  while (!hexahedra.empty()) {
    hexItr = removeHex(hexItr);
  }
}
// setVertexOrderIndices()===========
void sweetMesh::hexMesh::setVertexOrderIndices() {
  std::list<hexVertex>::iterator vertexItr;
  unsigned int n = 0;
  for (vertexItr = vertices.begin(); vertexItr != vertices.end();
       vertexItr++) {
    vertexItr->orderIndex = n++;
  }
}
// getSurfaceQuads()=================
void sweetMesh::hexMesh::getSurfaceQuads(std::list<quadFace> &surfaceQuads) {
  surfaceQuads.clear();
  for (std::list<quadFace>::iterator quadItr = quads.begin();
       quadItr != quads.end(); quadItr++) {
    if (quadItr->isSurfaceQuad) {
      surfaceQuads.push_back(*quadItr);
    }
  }
}
// computeAllHexJacobians()==========
void sweetMesh::hexMesh::computeAllHexJacobians() {
  numNonPosHexJacobians = 0;
  minHexJacobian = maxHexJacobian = 0.0;
  for (std::list<hexahedron>::iterator hexItr = hexahedra.begin();
       hexItr != hexahedra.end(); hexItr++) {
    hexItr->computeJacobians(minHexJacobian, maxHexJacobian,
                             numNonPosHexJacobians);
  }
}
// print()===========================
void sweetMesh::hexMesh::print() {
  std::cout << "Printing entire "
               "mesh.............................................\n";
  printHexes();
  std::cout << "Printing "
               "edges...................................................\n";
  printEdges();
  std::cout << "\nPrinting "
               "vertices..............................................\n";
  printVertices();
  std::cout << "\nPrinting "
               "hexCorners............................................\n";
  printHexCorners();
  std::cout << "Mesh has " << hexahedra.size() << " hexahedra;\t"
            << edges.size() << " edges;\t" << hexCorners.size()
            << " hexCorners;\t" << vertices.size() << " vertices\n";
}
// printStatistics()=================
void sweetMesh::hexMesh::printStatistics() {
  std::cout << "\nNumber of hexahedra:" << hexahedra.size()
            << "\nNumber of vertices: " << vertices.size()
            << "\nNumber of edges: " << edges.size()
            << "\nTotal # non-positive Jacobians = " << numNonPosHexJacobians
            << "\nGlobal minimal jacobian = " << minHexJacobian
            << "\nGlobal maximal jacobian = " << maxHexJacobian << "\n";
}
// printHexCorners()=================
void sweetMesh::hexMesh::printHexCorners() {
  for (std::list<hexCorner>::iterator cornerItr = hexCorners.begin();
       cornerItr != hexCorners.end(); cornerItr++) {
    cornerItr->print();
  }
}
// printVertices()===================
void sweetMesh::hexMesh::printVertices() {
  for (std::list<hexVertex>::iterator vertexItr = vertices.begin();
       vertexItr != vertices.end(); vertexItr++) {
    vertexItr->print();
  }
}
// printEdges()======================
void sweetMesh::hexMesh::printEdges() {
  for (std::list<hexEdge>::iterator edgeItr = edges.begin();
       edgeItr != edges.end(); edgeItr++) {
    edgeItr->print();
  }
}
// printQuads()======================
void sweetMesh::hexMesh::printQuads() {
  for (std::list<quadFace>::iterator quadItr = quads.begin();
       quadItr != quads.end(); quadItr++) {
    quadItr->print();
    std::cout << "\n";
  }
}
// printHexes()======================
void sweetMesh::hexMesh::printHexes() {
  for (std::list<hexahedron>::iterator hexItr = hexahedra.begin();
       hexItr != hexahedra.end(); hexItr++) {
    hexItr->print();
    std::cout << "\n";
  }
}
// addVertex()=======================
std::list<sweetMesh::hexVertex>::iterator
sweetMesh::hexMesh::addVertex(hexVertex &thisVertex) {
  std::list<hexVertex>::iterator vertexItr = vertices.begin();
  while (vertexItr != vertices.end() && (*vertexItr) <= thisVertex) {
    if (*vertexItr == thisVertex) {
      return vertexItr;
    }
    vertexItr++;
  }
  thisVertex.displayVertex = false;
  return vertices.insert(vertexItr, thisVertex);
}
// createEdges()=====================
void sweetMesh::hexMesh::createEdges(
    std::list<hexahedron>::iterator &hexItr) {
  addEdge(hexItr->cornerItrs[0]->myVertexItr,
          hexItr->cornerItrs[1]->myVertexItr, hexItr); // edge 0
  addEdge(hexItr->cornerItrs[1]->myVertexItr,
          hexItr->cornerItrs[2]->myVertexItr, hexItr); // edge 1
  addEdge(hexItr->cornerItrs[2]->myVertexItr,
          hexItr->cornerItrs[3]->myVertexItr, hexItr); // edge 2
  addEdge(hexItr->cornerItrs[3]->myVertexItr,
          hexItr->cornerItrs[0]->myVertexItr, hexItr); // edge 3
  addEdge(hexItr->cornerItrs[0]->myVertexItr,
          hexItr->cornerItrs[4]->myVertexItr, hexItr); // edge 4
  addEdge(hexItr->cornerItrs[1]->myVertexItr,
          hexItr->cornerItrs[5]->myVertexItr, hexItr); // edge 5
  addEdge(hexItr->cornerItrs[2]->myVertexItr,
          hexItr->cornerItrs[6]->myVertexItr, hexItr); // edge 6
  addEdge(hexItr->cornerItrs[3]->myVertexItr,
          hexItr->cornerItrs[7]->myVertexItr, hexItr); // edge 7
  addEdge(hexItr->cornerItrs[4]->myVertexItr,
          hexItr->cornerItrs[5]->myVertexItr, hexItr); // edge 8
  addEdge(hexItr->cornerItrs[5]->myVertexItr,
          hexItr->cornerItrs[6]->myVertexItr, hexItr); // edge 9
  addEdge(hexItr->cornerItrs[6]->myVertexItr,
          hexItr->cornerItrs[7]->myVertexItr, hexItr); // edge 10
  addEdge(hexItr->cornerItrs[7]->myVertexItr,
          hexItr->cornerItrs[4]->myVertexItr, hexItr); // edge 11
}
// addEdge()=========================
void sweetMesh::hexMesh::addEdge(std::list<hexVertex>::iterator &vA_Itr,
                                 std::list<hexVertex>::iterator &vB_Itr,
                                 std::list<hexahedron>::iterator &hex) {
  hexEdge newEdge(vA_Itr, vB_Itr);
  bool edgeAlreadyAdded = false;
  for (std::list<std::list<hexEdge>::iterator>::iterator adjacentEdgeItrItr =
           vA_Itr->adjacentEdgeItrs.begin();
       adjacentEdgeItrItr != vA_Itr->adjacentEdgeItrs.end() &&
       edgeAlreadyAdded == false;
       adjacentEdgeItrItr++) {
    if (**adjacentEdgeItrItr == newEdge) {
      edgeAlreadyAdded = true;
      (*adjacentEdgeItrItr)->adjacentHexItrs.push_back(hex);
      hex->adjacentEdges.push_back(*adjacentEdgeItrItr);
    }
  }
  if (edgeAlreadyAdded == false) {
    newEdge.adjacentHexItrs.push_back(hex);
    edges.push_back(newEdge);
    std::list<hexEdge>::iterator thisItr;
    thisItr = edges.end();
    thisItr--;
    vA_Itr->adjacentEdgeItrs.push_back(thisItr);
    vB_Itr->adjacentEdgeItrs.push_back(thisItr);
    hex->adjacentEdges.push_back(thisItr);
  }
}
// createQuads()=====================
void sweetMesh::hexMesh::createQuads(
    std::list<hexahedron>::iterator &hexItr) {
  // quadFace 0
  addQuad(hexItr->cornerItrs[0]->myVertexItr,
          hexItr->cornerItrs[1]->myVertexItr,
          hexItr->cornerItrs[2]->myVertexItr,
          hexItr->cornerItrs[3]->myVertexItr, hexItr, 0); // quadFace 0
  // quadFace 1
  addQuad(hexItr->cornerItrs[0]->myVertexItr,
          hexItr->cornerItrs[4]->myVertexItr,
          hexItr->cornerItrs[5]->myVertexItr,
          hexItr->cornerItrs[1]->myVertexItr, hexItr, 1); // quadFace 1
  // quadFace 2
  addQuad(hexItr->cornerItrs[1]->myVertexItr,
          hexItr->cornerItrs[5]->myVertexItr,
          hexItr->cornerItrs[6]->myVertexItr,
          hexItr->cornerItrs[2]->myVertexItr, hexItr, 2); // quadFace 2
  // quadFace 3
  addQuad(hexItr->cornerItrs[2]->myVertexItr,
          hexItr->cornerItrs[6]->myVertexItr,
          hexItr->cornerItrs[7]->myVertexItr,
          hexItr->cornerItrs[3]->myVertexItr, hexItr, 3); // quadFace 3
  // quadFace 4
  addQuad(hexItr->cornerItrs[0]->myVertexItr,
          hexItr->cornerItrs[3]->myVertexItr,
          hexItr->cornerItrs[7]->myVertexItr,
          hexItr->cornerItrs[4]->myVertexItr, hexItr, 4); // quadFace 4
  // quadFace 5
  addQuad(hexItr->cornerItrs[7]->myVertexItr,
          hexItr->cornerItrs[6]->myVertexItr,
          hexItr->cornerItrs[5]->myVertexItr,
          hexItr->cornerItrs[4]->myVertexItr, hexItr, 5); // quadFace 5
}
// addQuad()=========================
void sweetMesh::hexMesh::addQuad(std::list<hexVertex>::iterator &vertexA_Itr,
                                 std::list<hexVertex>::iterator &vertexB_Itr,
                                 std::list<hexVertex>::iterator &vertexC_Itr,
                                 std::list<hexVertex>::iterator &vertexD_Itr,
                                 std::list<hexahedron>::iterator &hexItr,
                                 unsigned int facePosition) {
  quadFace newQuad(vertexA_Itr, vertexB_Itr, vertexC_Itr, vertexD_Itr);
  // if the quad is already added then it must lie on the surface or there is
  // an error.
  bool quadAlreadyAdded = false;
  // TODO: change this to iterator just through the list of quads attached to
  // vertexA instead of the entire list of all quads
  for (std::list<quadFace>::iterator quadItr = quads.begin();
       quadItr != quads.end() && !quadAlreadyAdded; quadItr++) {
    if (*quadItr == newQuad) {
      quadItr->isSurfaceQuad = false;
      quadItr->nbrHex2 = hexItr;
      hexItr->faces[facePosition] = quadItr;
      quadAlreadyAdded = true;
    }
  }
  if (quadAlreadyAdded == false) {
    newQuad.isSurfaceQuad = true;
    newQuad.nbrHex1 = hexItr;
    newQuad.nbrHex2 = hexahedra.end();
    quads.push_back(newQuad);
    std::list<quadFace>::iterator thisItr;
    thisItr = quads.end();
    thisItr--;
    hexItr->faces[facePosition] = thisItr;
  }
}
// detachHex()=======================
void sweetMesh::hexMesh::detachHex(hexahedron &hex) {
  // detach faces
  for (unsigned int n = 0; n < 6; n++) {
    if (hex.faces[n]->isSurfaceQuad) {
      quads.erase(hex.faces[n]);
    } else { // interior quad
      if (*hex.faces[n]->nbrHex1 == hex) {
        hex.faces[n]->nbrHex1 = hex.faces[n]->nbrHex2;
      }
      hex.faces[n]->nbrHex2 = hexahedra.end();
      hex.faces[n]->isSurfaceQuad = true;
    }
  }

  // detach edges
  std::vector<std::list<hexEdge>::iterator>::iterator edgeItrItr;
  for (edgeItrItr = hex.adjacentEdges.begin();
       edgeItrItr != hex.adjacentEdges.end(); edgeItrItr++) {
    // If the current edge has no other neighboring hexes then simply delete
    // it
    if ((*edgeItrItr)->adjacentHexItrs.size() <= 1) {
      detachEdge(**edgeItrItr);
      edges.erase(*edgeItrItr);
    } else {
      // else keep the edge intact but don't let it point back to this hex
      for (std::list<std::list<hexahedron>::iterator>::iterator
               adjacentHexItrItr = (*edgeItrItr)->adjacentHexItrs.begin();
           adjacentHexItrItr != (*edgeItrItr)->adjacentHexItrs.end();
           adjacentHexItrItr++) {
        if (**adjacentHexItrItr == hex) {
          (*edgeItrItr)->adjacentHexItrs.erase(adjacentHexItrItr);
          adjacentHexItrItr = (*edgeItrItr)->adjacentHexItrs.end();
          adjacentHexItrItr--;
        }
      }
    }
  }
  for (unsigned int n = 0; n < 8; n++) {
    if (hex.cornerItrs[n]->myVertexItr->hexCornerItrs.size() <= 1) {
      vertices.erase(hex.cornerItrs[n]->myVertexItr);
    } else {
      for (std::list<std::list<hexCorner>::iterator>::iterator cornerItrItr =
               hex.cornerItrs[n]->myVertexItr->hexCornerItrs.begin();
           cornerItrItr !=
           hex.cornerItrs[n]->myVertexItr->hexCornerItrs.end();
           cornerItrItr++) {
        if (*((*cornerItrItr)->myHexItr) == hex) {
          hex.cornerItrs[n]->myVertexItr->hexCornerItrs.erase(cornerItrItr);
          cornerItrItr = hex.cornerItrs[n]->myVertexItr->hexCornerItrs.end();
          cornerItrItr--;
        }
      }
    }
    hexCorners.erase(hex.cornerItrs[n]);
  }
}
// detachEdge()======================
void sweetMesh::hexMesh::detachEdge(hexEdge &edge) {
  std::list<std::list<hexEdge>::iterator>::iterator edgeItrItr;
  for (edgeItrItr = edge.vA_Itr->adjacentEdgeItrs.begin();
       edgeItrItr != edge.vA_Itr->adjacentEdgeItrs.end(); edgeItrItr++) {
    if (**edgeItrItr == edge) {
      edge.vA_Itr->adjacentEdgeItrs.erase(edgeItrItr);
      edgeItrItr = edge.vA_Itr->adjacentEdgeItrs.end();
      edgeItrItr--;
    }
  }
  for (edgeItrItr = edge.vB_Itr->adjacentEdgeItrs.begin();
       edgeItrItr != edge.vB_Itr->adjacentEdgeItrs.end(); edgeItrItr++) {
    if (**edgeItrItr == edge) {
      edge.vB_Itr->adjacentEdgeItrs.erase(edgeItrItr);
      edgeItrItr = edge.vB_Itr->adjacentEdgeItrs.end();
      edgeItrItr--;
    }
  }
}
