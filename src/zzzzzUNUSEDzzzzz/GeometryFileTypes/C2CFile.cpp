/*
  Copyright 2002-2003 The University of Texas at Austin
  
	Authors: John Wiggins <prok@ices.utexas.edu>
					 Anthony Thane <thanea@ices.utexas.edu>
	Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of Volume Rover.

  Volume Rover is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  Volume Rover is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Volume Rover; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

// C2CFile.cpp: implementation of the C2CFile class.
//
//////////////////////////////////////////////////////////////////////

#include <GeometryFileTypes/C2CFile.h>
#include <cvcraw_geometry/Geometry.h>
#include <stdio.h>
//#include <qfileinfo.h>

#include <c2c_codec/c2c_codec.h>
#include <vector>

using std::vector;

C2CFile C2CFile::ms_C2CFileRepresentative;

// utility
static void MyNormalizationOf3Vector(float n[])
{
	float  result;
	result = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
	if (result == 0.0) {
		return;
	}
	n[0] = n[0]/result;
	n[1] = n[1]/result;
	n[2] = n[2]/result;
}

static void MyCrossProduct2(float u[], float v[], float w[])
{
	w[0] = u[1]*v[2] - u[2]*v[1];
	w[1] = u[2]*v[0] - v[2]*u[0];
	w[2] = u[0]*v[1] - v[0]*u[1];
	//printf("%f %f %f \n", w[0], w[1], w[2]);
	MyNormalizationOf3Vector(w);
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

C2CFile::C2CFile()
{

}

C2CFile::~C2CFile()
{

}

Geometry* C2CFile::loadFile(const string& fileName)
{
	ContourGeom *con = NULL;

	int nverts, ntris;
	bool colors = false;
	float *faceNorms;
	Geometry *retGeom = new Geometry();
	int i;

	// decode the file
	con = decodeC2CFile(fileName.c_str(), colors);
	
	//printf("building geometry...\n");
	// grab all the useful data from the ContourGeom object
	nverts = con->getNVert();
	ntris = con->getNTri();
	retGeom->AllocateTris(nverts, ntris);
	faceNorms = (float *)malloc(sizeof(float) * ntris * 3);

	if (colors) {
		retGeom->AllocateTriVertColors();
	}

	// Geometry members:
	//float* m_TriVerts;
	//float* m_TriVertNormals;
	//float* m_TriVertColors;
	//unsigned int* m_Tris;

	for (int v=0; v<nverts; v++)
	{
		retGeom->m_TriVerts[v*3+0] = con->vert[v][0];
		retGeom->m_TriVerts[v*3+1] = con->vert[v][1];
		retGeom->m_TriVerts[v*3+2] = con->vert[v][2];

		if (colors)
		{
			retGeom->m_TriVertColors[v*3+0] = con->vcol[v][0];
			retGeom->m_TriVertColors[v*3+1] = con->vcol[v][1];
			retGeom->m_TriVertColors[v*3+2] = con->vcol[v][2];
		}
	}

	for (int t=0; t<ntris; t++)
	{
		retGeom->m_Tris[t*3+0] = con->tri[t][0];
		retGeom->m_Tris[t*3+1] = con->tri[t][1];
		retGeom->m_Tris[t*3+2] = con->tri[t][2];
		//printf("%d, %d, %d\n", tris[t*3+0], tris[t*3+1], tris[t*3+2]);
	}

	// generate the face normals
	for (i = 0; i < ntris; i++)
	{
		float p[3][3], nm[3];
		float p01[3], p12[3];
		
		for (int jj=0; jj<3; jj++)
		{
			int v = con->tri[i][jj];
			p[jj][0] = con->vert[v][0];
			p[jj][1] = con->vert[v][1];
			p[jj][2] = con->vert[v][2];
		}
		
		for (int k=0; k<3; k++)
		{
			p01[k] = p[1][k] - p[0][k];
			p12[k] = p[2][k] - p[1][k];
		}
		
		MyCrossProduct2(p01, p12, nm);
		
		faceNorms[i*3+0] = nm[0];
		faceNorms[i*3+1] = nm[1];
		faceNorms[i*3+2] = nm[2];
		
		// printf("(%.2f, %.2f, %.2f) ", nm[0], nm[1], nm[2]);
		// printf("\n");
	}

	//printf("num of verts: %d\n", nverts);
	//printf("num of tris: %d\n", ntris);

	// find neighbor triangles of each vertex
	vector<int> *vectorNeighbor = new vector<int>[nverts];

	for (i=0; i<ntris; i++)
		for (int j=0; j<3; j++) 
			vectorNeighbor[con->tri[i][j]].push_back(i);

	// calculate vertex normals
	for (i=0; i<nverts; i++) {
		float ti=0, tj=0, tk=0;
		int numNeighbor=vectorNeighbor[i].size();

		for (int j=0; j<numNeighbor; j++) {
			ti += faceNorms[vectorNeighbor[i][j]*3+0];
			tj += faceNorms[vectorNeighbor[i][j]*3+1];
			tk += faceNorms[vectorNeighbor[i][j]*3+2];
			//printf("%f, %f, %f\n", ti, tj, tk);
		}

		retGeom->m_TriVertNormals[i*3+0]=(ti/(float)numNeighbor);
		retGeom->m_TriVertNormals[i*3+1]=(tj/(float)numNeighbor);
		retGeom->m_TriVertNormals[i*3+2]=(tk/(float)numNeighbor);
		//printf("vertex#: %d N(%f, %f, %f)\n", i, norms[i*3+0], norms[i*3+1], norms[i*3+2]);
	}

	free(faceNorms);
	delete [] vectorNeighbor;
	// delete the ContourGeom
	delete con; con=0;

	return retGeom;
}

// determine if a file is a c2c file
bool C2CFile::checkType(const string& fileName)
{
	ContourGeom *con = NULL;
	bool ret = false, dummy;

	// decode the file
	con = decodeC2CFile(fileName.c_str(), dummy);

	// this is a valid file if con is not null and has at least one vertex
	ret = (con && con->getNVert() != 0);

	// don't leak
	delete con;

	return ret;
}

// save a geometry instance to a c2c file
bool C2CFile::saveFile(const Geometry* geometry, const string& fileName)
{
	// saving to c2c is not supported right now
	return false;
}

GeometryFileType* C2CFile::getRepresentative()
{
	return &ms_C2CFileRepresentative;
}

