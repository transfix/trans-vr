/***************************************************************************
 *   Copyright (C) 2009 by Bharadwaj Subramanian   *
 *   bharadwajs@pupil.ices.utexas.edu   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#ifndef __MESH_H
#define __MESH_H

#include "Grid.h"
#include <VolMagick/VolMagick.h>
#include "MCTester.h"
#include <vector>
#include <map>
#include <string>

using namespace std;

struct BoundingBox
{
  float minX,minY,minZ,maxX,maxY,maxZ;
};

class Mesh
{
  private:
    string getVertexHash(Point p);
    string getTriangleHash(vector<unsigned int> triangle);
    map<string,unsigned int> vertexMap;
    map<string,unsigned int> triangleMap;
    Point computeNormal(Point &v1,Point &v2,Point &v3);
    float dotProduct(Point &v1,Point &v2);
    string edgeHash(unsigned int v1, unsigned int v2);
  public:
    Mesh();
    Mesh(float color[3]);
    vector<Point> vertices;
    float vertColors[3];
    BoundingBox bb;
    vector< vector<unsigned int> > triangles;
    unsigned int addVertex(Point &p,unsigned int color);
    unsigned int addTriangle(vector<unsigned int> &t);
    unsigned int addTriangle(unsigned int i,unsigned int j,unsigned int k);
    unsigned int getNumVertices(void) { return vertices.size(); }
    unsigned int getNumTriangles(void) { return triangles.size(); }
    bool triangleExists(unsigned int i,unsigned int j,unsigned int k);
    bool reorientMesh(void);
    void clearMesh(void);
    void saveMesh(string filename,VolMagick::Volume *v);
    void saveMesh(string filename);
    void correctNormals(void);
    unsigned int getColor(MCCase mcCase); 
    void computeBoundingBox(void);
};
#endif
