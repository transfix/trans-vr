#ifndef HEXMESH_H
#define HEXMESH_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <list>
#include <queue>

#include <boost/any.hpp>

#include <SweetMesh/vertex.h>
#include <SweetMesh/quad.h>
#include <SweetMesh/hexahedron.h>

namespace sweetMesh{

//===============================================================================
//hexMesh
//===============================================================================
class hexMesh{
public:
	std::list<hexVertex>		vertices;
	std::list<hexCorner>		hexCorners;
	std::list<hexEdge>		edges;
	std::list<quadFace>		quads;
	std::list<hexahedron>		hexahedra;
	double 					minHexJacobian,	maxHexJacobian;
	unsigned int				numNonPosHexJacobians;
	std::queue<unsigned long long>		deletedHandles;

	hexMesh() { minHexJacobian = maxHexJacobian = 0.0; numNonPosHexJacobians = 0; }
	~hexMesh() { }
	hexMesh* getPointer() { return this; }

	std::list<hexahedron>::iterator addHex(hexVertex& V0, hexVertex& V1, hexVertex& V2, hexVertex& V3, hexVertex& V4, hexVertex& V5, hexVertex& V6, hexVertex& V7, unsigned int orderIndex=0);
	std::list<hexahedron>::iterator	removeHex(std::list<hexahedron>::iterator& deleteHexItr);
	void clear();

	void setVertexOrderIndices();
	void getSurfaceQuads(std::list<quadFace>& surfaceQuads);
	void computeAllHexJacobians();

	void print();
	void printStatistics();
	void printHexCorners();
	void printVertices();
	void printEdges();
	void printQuads();
	void printHexes();

	void refreshDisplay();
	void clearDisplay();
	void displayAllEdges();
	void displayAllSurfaceQuads();

protected:
	std::list<hexVertex>::iterator addVertex(hexVertex& thisVertex);
	void createEdges(std::list<hexahedron>::iterator& hexItr);
	void addEdge(std::list<hexVertex>::iterator& vertexA_Itr, std::list<hexVertex>::iterator& vertexB_Itr, std::list<hexahedron>::iterator& hexItr);
	void createQuads(std::list<hexahedron>::iterator& hexItr);
	void addQuad(std::list<hexVertex>::iterator& vertexA_Itr, std::list<hexVertex>::iterator& vertexB_Itr, std::list<hexVertex>::iterator& vertexC_Itr, std::list<hexVertex>::iterator& vertexD_Itr, std::list<hexahedron>::iterator& hexItr, unsigned int facePosition);
	void detachHex(hexahedron& hex);
	void detachEdge(hexEdge& edge);
};

}

#endif
